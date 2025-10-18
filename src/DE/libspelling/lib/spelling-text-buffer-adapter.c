/*
 * spelling-text-buffer-adapter.c
 *
 * Copyright 2021-2023 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "egg-action-group.h"

#include "spelling-compat-private.h"
#include "spelling-checker-private.h"
#include "spelling-cursor-private.h"
#include "spelling-engine-private.h"
#include "spelling-menu-private.h"
#include "spelling-text-buffer-adapter.h"

#define NO_SPELL_CHECK_TAG "gtksourceview:context-classes:no-spell-check"

/**
 * SpellingTextBufferAdapter:
 *
 * `SpellingTextBufferAdapter` implements helpers to easily add spellchecking
 * capabilities to a `GtkSourceBuffer`.
 */

#define INVALIDATE_DELAY_MSECS 100
#define MAX_WORD_CHARS 100

struct _SpellingTextBufferAdapter
{
  GObject          parent_instance;

  SpellingEngine  *engine;
  GSignalGroup    *buffer_signals;
  GWeakRef         buffer_wr;
  SpellingChecker *checker;
  GtkTextTag      *no_spell_check_tag;
  GMenuModel      *menu;
  GMenu           *top_menu;
  char            *word_under_cursor;

  /* Borrowed pointers */
  GtkTextMark     *insert_mark;
  GtkTextTag      *tag;

  guint            commit_handler;

  guint            cursor_position;
  guint            incoming_cursor_position;
  guint            queued_cursor_moved;

  guint            enabled : 1;
};

static void spelling_add_action      (SpellingTextBufferAdapter *self,
                                      GVariant                  *param);
static void spelling_ignore_action   (SpellingTextBufferAdapter *self,
                                      GVariant                  *param);
static void spelling_enabled_action  (SpellingTextBufferAdapter *self,
                                      GVariant                  *param);
static void spelling_correct_action  (SpellingTextBufferAdapter *self,
                                      GVariant                  *param);
static void spelling_language_action (SpellingTextBufferAdapter *self,
                                      GVariant                  *param);

EGG_DEFINE_ACTION_GROUP (SpellingTextBufferAdapter, spelling_text_buffer_adapter, {
  { "add", spelling_add_action },
  { "correct", spelling_correct_action, "s" },
  { "enabled", spelling_enabled_action, NULL, "false" },
  { "ignore", spelling_ignore_action },
  { "language", spelling_language_action, "s", "''" },
})

G_DEFINE_FINAL_TYPE_WITH_CODE (SpellingTextBufferAdapter, spelling_text_buffer_adapter, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_ACTION_GROUP, spelling_text_buffer_adapter_init_action_group))

enum {
  PROP_0,
  PROP_BUFFER,
  PROP_CHECKER,
  PROP_ENABLED,
  PROP_LANGUAGE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
spelling_text_buffer_adapter_commit_notify (GtkTextBuffer            *buffer,
                                            GtkTextBufferNotifyFlags  flags,
                                            guint                     position,
                                            guint                     length,
                                            gpointer                  user_data)
{
  SpellingTextBufferAdapter *self = user_data;

  g_assert (GTK_IS_TEXT_BUFFER (buffer));
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  if (flags == GTK_TEXT_BUFFER_NOTIFY_BEFORE_INSERT)
    spelling_engine_before_insert_text (self->engine, position, length);
  else if (flags == GTK_TEXT_BUFFER_NOTIFY_AFTER_INSERT)
    spelling_engine_after_insert_text (self->engine, position, length);
  else if (flags == GTK_TEXT_BUFFER_NOTIFY_BEFORE_DELETE)
    spelling_engine_before_delete_range (self->engine, position, length);
  else if (flags == GTK_TEXT_BUFFER_NOTIFY_AFTER_DELETE)
    spelling_engine_after_delete_range (self->engine, position);
}

static gboolean
spelling_text_buffer_adapter_check_enabled (gpointer instance)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return FALSE;

  if (gtk_source_buffer_get_loading (GTK_SOURCE_BUFFER (buffer)))
    return FALSE;

  return self->enabled;
}

static guint
spelling_text_buffer_adapter_get_cursor (gpointer instance)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  GtkTextIter iter;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return 0;

  gtk_text_buffer_get_iter_at_mark (buffer, &iter, self->insert_mark);

  return gtk_text_iter_get_offset (&iter);
}

static char *
spelling_text_buffer_adapter_copy_text (gpointer instance,
                                        guint    position,
                                        guint    length)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  GtkTextIter begin;
  GtkTextIter end;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    {
      g_warn_if_reached ();
      return g_new0 (char, length + 1);
    }

  gtk_text_buffer_get_iter_at_offset (buffer, &begin, position);
  gtk_text_buffer_get_iter_at_offset (buffer, &end, position + length);

  return gtk_text_iter_get_slice (&begin, &end);
}

static void
spelling_text_buffer_adapter_apply_tag (gpointer instance,
                                        guint    position,
                                        guint    length)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  GtkTextIter begin;
  GtkTextIter end;

  if (self->tag == NULL)
    return;

  /* If the position overlaps our cursor position, ignore it. We don't
   * want to show that to the user while they are typing and will
   * instead deal with it when the cursor leaves the word.
   */
  if (position <= self->cursor_position &&
      position + length >= self->cursor_position)
    return;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return;

  gtk_text_buffer_get_iter_at_offset (buffer, &begin, position);
  gtk_text_buffer_get_iter_at_offset (buffer, &end, position + length);
  gtk_text_buffer_apply_tag (buffer, self->tag, &begin, &end);
}

static void
spelling_text_buffer_adapter_clear_tag (gpointer instance,
                                        guint    position,
                                        guint    length)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  GtkTextIter begin;
  GtkTextIter end;

  if (self->tag == NULL)
    return;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return;

  gtk_text_buffer_get_iter_at_offset (buffer, &begin, position);
  gtk_text_buffer_get_iter_at_offset (buffer, &end, position + length);
  gtk_text_buffer_remove_tag (buffer, self->tag, &begin, &end);
}

static gboolean
spelling_text_buffer_adapter_backward_word_start (gpointer  instance,
                                                  guint    *position)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  const char *extra_word_chars = NULL;
  GtkTextIter iter;
  guint prev = *position;

  if (self->checker != NULL)
    extra_word_chars = spelling_checker_get_extra_word_chars (self->checker);

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return FALSE;

  gtk_text_buffer_get_iter_at_offset (buffer, &iter, *position);

  spelling_iter_backward_word_start (&iter, extra_word_chars);

  *position = gtk_text_iter_get_offset (&iter);

  return prev != *position;
}

static gboolean
spelling_text_buffer_adapter_forward_word_end (gpointer  instance,
                                               guint    *position)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  const char *extra_word_chars = NULL;
  GtkTextIter iter;
  guint prev = *position;

  if (self->checker != NULL)
    extra_word_chars = spelling_checker_get_extra_word_chars (self->checker);

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return FALSE;

  gtk_text_buffer_get_iter_at_offset (buffer, &iter, *position);

  spelling_iter_forward_word_end (&iter, extra_word_chars);

  *position = gtk_text_iter_get_offset (&iter);

  return prev != *position;
}

static PangoLanguage *
spelling_text_buffer_adapter_get_pango_language (gpointer instance)
{
  SpellingTextBufferAdapter *self = instance;

  return _spelling_checker_get_pango_language (self->checker);
}

static SpellingDictionary *
spelling_text_buffer_adapter_get_dictionary (gpointer instance)
{
  SpellingTextBufferAdapter *self = instance;

  return _spelling_checker_get_dictionary (self->checker);
}

static void
spelling_text_buffer_adapter_intersect_spellcheck_region (gpointer   instance,
                                                          GtkBitset *region)
{
  SpellingTextBufferAdapter *self = instance;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  GtkTextIter begin;
  GtkTextIter end;
  GtkTextIter iter;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  if (self->no_spell_check_tag == NULL || gtk_bitset_is_empty (region))
    return;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return;

  gtk_text_buffer_get_iter_at_offset (buffer,
                                      &begin,
                                      gtk_bitset_get_minimum (region));
  gtk_text_buffer_get_iter_at_offset (buffer,
                                      &end,
                                      gtk_bitset_get_maximum (region));

  if (gtk_text_iter_has_tag (&begin, self->no_spell_check_tag))
    {
      if (!gtk_text_iter_starts_tag (&begin, self->no_spell_check_tag))
        gtk_text_iter_backward_to_tag_toggle (&begin, self->no_spell_check_tag);
    }
  else
    {
      gtk_text_iter_forward_to_tag_toggle (&begin, self->no_spell_check_tag);
    }

  /* At this point we either have a no-spell-check tag or the
   * @begin iter will be at the end of the file and we can be
   * certain it will be >= 0.
   */
  while (gtk_text_iter_compare (&begin, &end) < 0)
    {
      iter = begin;

      gtk_text_iter_forward_to_tag_toggle (&iter, self->no_spell_check_tag);

      g_assert (gtk_text_iter_compare (&begin, &iter) < 0);
      g_assert (gtk_text_iter_has_tag (&begin, self->no_spell_check_tag));
      g_assert (!gtk_text_iter_has_tag (&iter, self->no_spell_check_tag));

#if 0
      g_print ("%u:%u to %u:%u: NO SPELL CHECK (%u -> %u)\n",
               gtk_text_iter_get_line (&begin) + 1,
               gtk_text_iter_get_line_offset (&begin) + 1,
               gtk_text_iter_get_line (&iter) + 1,
               gtk_text_iter_get_line_offset (&iter) + 1,
               gtk_text_iter_get_offset (&begin),
               gtk_text_iter_get_offset (&iter));
#endif

      gtk_bitset_remove_range_closed (region,
                                      gtk_text_iter_get_offset (&begin),
                                      gtk_text_iter_get_offset (&iter) - 1);

      begin = iter;

      gtk_text_iter_forward_to_tag_toggle (&begin, self->no_spell_check_tag);
    }
}

static const SpellingAdapter adapter_funcs = {
  .check_enabled = spelling_text_buffer_adapter_check_enabled,
  .get_cursor = spelling_text_buffer_adapter_get_cursor,
  .copy_text = spelling_text_buffer_adapter_copy_text,
  .apply_tag = spelling_text_buffer_adapter_apply_tag,
  .clear_tag = spelling_text_buffer_adapter_clear_tag,
  .backward_word_start = spelling_text_buffer_adapter_backward_word_start,
  .forward_word_end = spelling_text_buffer_adapter_forward_word_end,
  .get_language = spelling_text_buffer_adapter_get_pango_language,
  .get_dictionary = spelling_text_buffer_adapter_get_dictionary,
  .intersect_spellcheck_region = spelling_text_buffer_adapter_intersect_spellcheck_region,
};

static inline gboolean
forward_word_end (SpellingTextBufferAdapter *self,
                  GtkTextIter               *iter)
{
  const char *extra_word_chars = NULL;

  if (self->checker != NULL)
    extra_word_chars = spelling_checker_get_extra_word_chars (self->checker);

  return spelling_iter_forward_word_end (iter, extra_word_chars);
}

static inline gboolean
backward_word_start (SpellingTextBufferAdapter *self,
                     GtkTextIter               *iter)
{
  const char *extra_word_chars = NULL;

  if (self->checker != NULL)
    extra_word_chars = spelling_checker_get_extra_word_chars (self->checker);

  return spelling_iter_backward_word_start (iter, extra_word_chars);
}

static gboolean
get_word_at_position (SpellingTextBufferAdapter *self,
                      guint                      position,
                      GtkTextIter               *begin,
                      GtkTextIter               *end)
{
  g_autoptr(GtkTextBuffer) buffer = NULL;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return FALSE;

  gtk_text_buffer_get_iter_at_offset (buffer, begin, position);
  *end = *begin;

  if (gtk_text_iter_ends_word (end))
    {
      backward_word_start (self, begin);
      return TRUE;
    }

  if (!gtk_text_iter_starts_word (begin))
    {
      if (!gtk_text_iter_inside_word (begin))
        return FALSE;

      backward_word_start (self, begin);
    }

  if (!gtk_text_iter_ends_word (end))
    forward_word_end (self, end);

  return TRUE;
}

/**
 * spelling_text_buffer_adapter_new:
 * @buffer: (not nullable): a `GtkSourceBuffer`
 * @checker: a `SpellingChecker`
 *
 * Create a new `SpellingTextBufferAdapter`.
 *
 * Returns: (transfer full): a newly created `SpellingTextBufferAdapter`
 */
SpellingTextBufferAdapter *
spelling_text_buffer_adapter_new (GtkSourceBuffer *buffer,
                                  SpellingChecker *checker)
{
  g_return_val_if_fail (GTK_SOURCE_IS_BUFFER (buffer), NULL);
  g_return_val_if_fail (!checker || SPELLING_IS_CHECKER (checker), NULL);

  return g_object_new (SPELLING_TYPE_TEXT_BUFFER_ADAPTER,
                       "buffer", buffer,
                       "checker", checker,
                       NULL);
}

/**
 * spelling_text_buffer_adapter_invalidate_all:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Invalidate the spelling engine, to force parsing again.
 *
 * Invalidation is automatically done on [property@GtkSource.Buffer:loading]
 * change.
 */
void
spelling_text_buffer_adapter_invalidate_all (SpellingTextBufferAdapter *self)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  spelling_engine_invalidate_all (self->engine);
}

static void
on_tag_added_cb (SpellingTextBufferAdapter *self,
                 GtkTextTag                *tag,
                 GtkTextTagTable           *tag_table)
{
  g_autofree char *name = NULL;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (GTK_IS_TEXT_TAG (tag));
  g_assert (GTK_IS_TEXT_TAG_TABLE (tag_table));

  g_object_get (tag,
                "name", &name,
                NULL);

  if (name && strcmp (name, NO_SPELL_CHECK_TAG) == 0)
    {
      g_set_object (&self->no_spell_check_tag, tag);
      spelling_text_buffer_adapter_invalidate_all (self);
    }
}

static void
on_tag_removed_cb (SpellingTextBufferAdapter *self,
                   GtkTextTag                *tag,
                   GtkTextTagTable           *tag_table)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (GTK_IS_TEXT_TAG (tag));
  g_assert (GTK_IS_TEXT_TAG_TABLE (tag_table));

  if (tag == self->no_spell_check_tag)
    {
      g_clear_object (&self->no_spell_check_tag);
      spelling_text_buffer_adapter_invalidate_all (self);
    }
}

static void
invalidate_tag_region_cb (SpellingTextBufferAdapter *self,
                          GtkTextTag                *tag,
                          GtkTextIter               *begin,
                          GtkTextIter               *end,
                          GtkTextBuffer             *buffer)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (GTK_IS_TEXT_TAG (tag));
  g_assert (GTK_IS_TEXT_BUFFER (buffer));

  if (tag == self->no_spell_check_tag)
    {
      gtk_text_iter_order (begin, end);
      spelling_engine_invalidate (self->engine,
                                  gtk_text_iter_get_offset (begin),
                                  gtk_text_iter_get_offset (end) - gtk_text_iter_get_offset (begin));
    }
}

static void
apply_error_style_cb (GtkSourceBuffer *buffer,
                      GParamSpec      *pspec,
                      GtkTextTag      *tag)
{
  GtkSourceStyleScheme *scheme;
  GtkSourceStyle *style;
  static GdkRGBA error_color;

  g_assert (GTK_SOURCE_IS_BUFFER (buffer));
  g_assert (GTK_IS_TEXT_TAG (tag));

  if G_UNLIKELY (error_color.alpha == .0)
    gdk_rgba_parse (&error_color, "#e01b24");

  g_object_set (tag,
                "underline", PANGO_UNDERLINE_ERROR_LINE,
                "underline-rgba", &error_color,
                "background-set", FALSE,
                "foreground-set", FALSE,
                "weight-set", FALSE,
                "variant-set", FALSE,
                "style-set", FALSE,
                "indent-set", FALSE,
                "size-set", FALSE,
                NULL);

  if ((scheme = gtk_source_buffer_get_style_scheme (buffer)))
    {
      if ((style = gtk_source_style_scheme_get_style (scheme, "def:misspelled-word")))
        gtk_source_style_apply (style, tag);
    }
}

static void
spelling_text_buffer_adapter_set_buffer (SpellingTextBufferAdapter *self,
                                         GtkSourceBuffer           *buffer)
{
  GtkTextIter begin, end;
  GtkTextTagTable *tag_table;
  GtkTextTag *tag;
  guint offset;
  guint length;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (GTK_SOURCE_IS_BUFFER (buffer));

  g_weak_ref_set (&self->buffer_wr, buffer);

  self->insert_mark = gtk_text_buffer_get_insert (GTK_TEXT_BUFFER (buffer));

  self->commit_handler =
    gtk_text_buffer_add_commit_notify (GTK_TEXT_BUFFER (buffer),
                                       (GTK_TEXT_BUFFER_NOTIFY_BEFORE_INSERT |
                                        GTK_TEXT_BUFFER_NOTIFY_AFTER_INSERT |
                                        GTK_TEXT_BUFFER_NOTIFY_BEFORE_DELETE |
                                        GTK_TEXT_BUFFER_NOTIFY_AFTER_DELETE),
                                       spelling_text_buffer_adapter_commit_notify,
                                       self, NULL);

  g_signal_group_set_target (self->buffer_signals, buffer);

  gtk_text_buffer_get_bounds (GTK_TEXT_BUFFER (buffer), &begin, &end);

  offset = gtk_text_iter_get_offset (&begin);
  length = gtk_text_iter_get_offset (&end) - offset;

  if (length > 0)
    {
      spelling_engine_before_insert_text (self->engine, offset, length);
      spelling_engine_after_insert_text (self->engine, offset, length);
    }

  self->tag = gtk_text_buffer_create_tag (GTK_TEXT_BUFFER (buffer), NULL,
                                          "underline", PANGO_UNDERLINE_ERROR,
                                          NULL);

  g_signal_connect_object (buffer,
                           "notify::style-scheme",
                           G_CALLBACK (apply_error_style_cb),
                           self->tag,
                           0);
  apply_error_style_cb (GTK_SOURCE_BUFFER (buffer), NULL, self->tag);

  /* Track tag changes from the tag table and extract "no-spell-check"
   * tag from GtkSourceView so that we can avoid words with that tag.
   */
  tag_table = gtk_text_buffer_get_tag_table (GTK_TEXT_BUFFER (buffer));
  g_signal_connect_object (tag_table,
                           "tag-added",
                           G_CALLBACK (on_tag_added_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (tag_table,
                           "tag-removed",
                           G_CALLBACK (on_tag_removed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (buffer,
                           "apply-tag",
                           G_CALLBACK (invalidate_tag_region_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (buffer,
                           "remove-tag",
                           G_CALLBACK (invalidate_tag_region_cb),
                           self,
                           G_CONNECT_SWAPPED);

  if ((tag = gtk_text_tag_table_lookup (tag_table, NO_SPELL_CHECK_TAG)))
    on_tag_added_cb (self, tag, tag_table);
}

static void
remember_word_under_cursor (SpellingTextBufferAdapter *self)
{
  g_autofree char *word = NULL;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  g_auto(GStrv) corrections = NULL;
  GtkTextMark *insert;
  GtkTextIter iter, begin, end;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  g_clear_pointer (&self->word_under_cursor, g_free);

  if (self->checker == NULL)
    goto cleanup;

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    goto cleanup;

  insert = gtk_text_buffer_get_insert (buffer);

  gtk_text_buffer_get_iter_at_mark (buffer, &iter, insert);

  if (get_word_at_position (self, gtk_text_iter_get_offset (&iter), &begin, &end))
    {
      word = gtk_text_iter_get_slice (&begin, &end);

      if (spelling_checker_check_word (self->checker, word, -1))
        g_clear_pointer (&word, g_free);
      else
        corrections = spelling_checker_list_corrections (self->checker, word);
    }

cleanup:
  g_set_str (&self->word_under_cursor, word);

  spelling_text_buffer_adapter_set_action_enabled (self, "add", !!word);
  spelling_text_buffer_adapter_set_action_enabled (self, "ignore", !!word);

  if (self->menu)
    spelling_menu_set_corrections (self->menu, word, (const char * const *)corrections);
}

/**
 * spelling_text_buffer_adapter_set_enabled:
 * @self: a `SpellingTextBufferAdapter`
 * @enabled: whether the spellcheck is enabled
 *
 * If %TRUE spellcheck is enabled.
 */
void
spelling_text_buffer_adapter_set_enabled (SpellingTextBufferAdapter *self,
                                          gboolean                   enabled)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  enabled = !!enabled;

  if (enabled != self->enabled)
    {
      self->enabled = enabled;
      spelling_text_buffer_adapter_set_action_state (self,
                                                     "enabled",
                                                     g_variant_new_boolean (enabled));

      if (!enabled)
        {
          spelling_text_buffer_adapter_set_action_enabled (self, "add", FALSE);
          spelling_text_buffer_adapter_set_action_enabled (self, "ignore", FALSE);

          if (self->menu)
            spelling_menu_set_corrections (self->menu, NULL, NULL);
        }
      else
        {
          remember_word_under_cursor (self);
        }

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENABLED]);
      spelling_engine_invalidate_all (self->engine);
    }
}

static gboolean
spelling_text_buffer_adapter_cursor_moved_cb (gpointer data)
{
  SpellingTextBufferAdapter *self = data;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  GtkTextIter begin, end;
  gboolean enabled;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  self->queued_cursor_moved = 0;

  /* Protect against weak-pointer lost */
  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return G_SOURCE_REMOVE;

  enabled = spelling_text_buffer_adapter_get_enabled (self);

  /* Invalidate the old position */
  if (enabled && get_word_at_position (self, self->cursor_position, &begin, &end))
    spelling_engine_invalidate (self->engine,
                                gtk_text_iter_get_offset (&begin),
                                gtk_text_iter_get_offset (&end) - gtk_text_iter_get_offset (&begin));

  self->cursor_position = self->incoming_cursor_position;

  /* Invalidate word at new position */
  if (enabled && get_word_at_position (self, self->cursor_position, &begin, &end))
    spelling_engine_invalidate (self->engine,
                                gtk_text_iter_get_offset (&begin),
                                gtk_text_iter_get_offset (&end) - gtk_text_iter_get_offset (&begin));

  remember_word_under_cursor (self);

  return G_SOURCE_REMOVE;
}

static void
spelling_text_buffer_adapter_cursor_moved (SpellingTextBufferAdapter *self,
                                           GtkSourceBuffer           *buffer)
{
  GtkTextIter iter;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (GTK_SOURCE_IS_BUFFER (buffer));

  gtk_text_buffer_get_iter_at_mark (GTK_TEXT_BUFFER (buffer), &iter, self->insert_mark);
  self->incoming_cursor_position = gtk_text_iter_get_offset (&iter);
  g_clear_handle_id (&self->queued_cursor_moved, g_source_remove);

  if (!spelling_text_buffer_adapter_check_enabled (self))
    return;

  self->queued_cursor_moved = g_timeout_add_full (G_PRIORITY_LOW,
                                                  INVALIDATE_DELAY_MSECS,
                                                  spelling_text_buffer_adapter_cursor_moved_cb,
                                                  g_object_ref (self),
                                                  g_object_unref);
}

static void
spelling_text_buffer_adapter_notify_loading_cb (SpellingTextBufferAdapter *self,
                                                GParamSpec                *pspec,
                                                GtkSourceBuffer           *buffer)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (GTK_SOURCE_IS_BUFFER (buffer));

  if (self->engine != NULL)
    spelling_engine_invalidate_all (self->engine);
}

static void
spelling_text_buffer_adapter_finalize (GObject *object)
{
  SpellingTextBufferAdapter *self = (SpellingTextBufferAdapter *)object;

  self->tag = NULL;
  self->insert_mark = NULL;

  g_clear_pointer (&self->word_under_cursor, g_free);
  g_clear_object (&self->checker);
  g_clear_object (&self->no_spell_check_tag);
  g_clear_object (&self->buffer_signals);
  g_weak_ref_clear (&self->buffer_wr);

  G_OBJECT_CLASS (spelling_text_buffer_adapter_parent_class)->finalize (object);
}

static void
spelling_text_buffer_adapter_dispose (GObject *object)
{
  SpellingTextBufferAdapter *self = (SpellingTextBufferAdapter *)object;
  g_autoptr(GtkTextBuffer) buffer = NULL;

  if ((buffer = g_weak_ref_get (&self->buffer_wr)))
    {
      gtk_text_buffer_remove_commit_notify (buffer, self->commit_handler);
      self->commit_handler = 0;
      g_weak_ref_set (&self->buffer_wr, NULL);
    }

  g_signal_group_set_target (self->buffer_signals, NULL);
  g_clear_object (&self->engine);
  g_clear_object (&self->menu);
  g_clear_object (&self->top_menu);

  G_OBJECT_CLASS (spelling_text_buffer_adapter_parent_class)->dispose (object);
}

static void
spelling_text_buffer_adapter_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  SpellingTextBufferAdapter *self = SPELLING_TEXT_BUFFER_ADAPTER (object);

  switch (prop_id)
    {
    case PROP_BUFFER:
      g_value_take_object (value, g_weak_ref_get (&self->buffer_wr));
      break;

    case PROP_CHECKER:
      g_value_set_object (value, spelling_text_buffer_adapter_get_checker (self));
      break;

    case PROP_ENABLED:
      g_value_set_boolean (value, spelling_text_buffer_adapter_get_enabled (self));
      break;

    case PROP_LANGUAGE:
      g_value_set_string (value, spelling_text_buffer_adapter_get_language (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_text_buffer_adapter_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  SpellingTextBufferAdapter *self = SPELLING_TEXT_BUFFER_ADAPTER (object);

  switch (prop_id)
    {
    case PROP_BUFFER:
      spelling_text_buffer_adapter_set_buffer (self, g_value_get_object (value));
      break;

    case PROP_CHECKER:
      spelling_text_buffer_adapter_set_checker (self, g_value_get_object (value));
      break;

    case PROP_ENABLED:
      spelling_text_buffer_adapter_set_enabled (self, g_value_get_boolean (value));
      break;

    case PROP_LANGUAGE:
      spelling_text_buffer_adapter_set_language (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_text_buffer_adapter_class_init (SpellingTextBufferAdapterClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = spelling_text_buffer_adapter_dispose;
  object_class->finalize = spelling_text_buffer_adapter_finalize;
  object_class->get_property = spelling_text_buffer_adapter_get_property;
  object_class->set_property = spelling_text_buffer_adapter_set_property;

  /**
   * SpellingTextBufferAdapter:buffer:
   *
   * The [class@GtkSource.Buffer].
   */
  properties[PROP_BUFFER] =
    g_param_spec_object ("buffer", NULL, NULL,
                         GTK_SOURCE_TYPE_BUFFER,
                         (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  /**
   * SpellingTextBufferAdapter:checker:
   *
   * The [class@Spelling.Checker].
   */
  properties[PROP_CHECKER] =
    g_param_spec_object ("checker", NULL, NULL,
                         SPELLING_TYPE_CHECKER,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * SpellingTextBufferAdapter:enabled:
   *
   * Whether spellcheck is enabled.
   */
  properties[PROP_ENABLED] =
    g_param_spec_boolean ("enabled", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  /**
   * SpellingTextBufferAdapter:language:
   *
   * The language code, such as `en_US`.
   */
  properties[PROP_LANGUAGE] =
    g_param_spec_string ("language", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
spelling_text_buffer_adapter_init (SpellingTextBufferAdapter *self)
{
  g_weak_ref_init (&self->buffer_wr, NULL);

  self->enabled = TRUE;
  spelling_text_buffer_adapter_set_action_state (self,
                                                 "enabled",
                                                 g_variant_new_boolean (TRUE));

  self->buffer_signals = g_signal_group_new (GTK_SOURCE_TYPE_BUFFER);

  g_signal_group_connect_object (self->buffer_signals,
                                 "cursor-moved",
                                 G_CALLBACK (spelling_text_buffer_adapter_cursor_moved),
                                 self,
                                 G_CONNECT_SWAPPED);
  g_signal_group_connect_object (self->buffer_signals,
                                 "notify::loading",
                                 G_CALLBACK (spelling_text_buffer_adapter_notify_loading_cb),
                                 self,
                                 G_CONNECT_SWAPPED);

  self->engine = spelling_engine_new (&adapter_funcs, G_OBJECT (self));
}

/**
 * spelling_text_buffer_adapter_get_checker:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Gets the checker used by the adapter.
 *
 * Returns: (transfer none) (nullable): a `SpellingChecker` or %NULL
 */
SpellingChecker *
spelling_text_buffer_adapter_get_checker (SpellingTextBufferAdapter *self)
{
  g_return_val_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self), NULL);

  return self->checker;
}

static void
spelling_text_buffer_adapter_checker_notify_language (SpellingTextBufferAdapter *self,
                                                      GParamSpec                *pspec,
                                                      SpellingChecker           *checker)
{
  const char *code;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (SPELLING_IS_CHECKER (checker));

  if (!(code = spelling_checker_get_language (checker)))
    code = "";

  spelling_text_buffer_adapter_set_action_state (self, "language", g_variant_new_string (code));
}

/**
 * spelling_text_buffer_adapter_set_checker:
 * @self: a `SpellingTextBufferAdapter`
 * @checker: a `SpellingChecker`
 *
 * Set the [class@Spelling.Checker] used for spellchecking.
 */
void
spelling_text_buffer_adapter_set_checker (SpellingTextBufferAdapter *self,
                                          SpellingChecker           *checker)
{
  const char *code = "";

  g_return_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_return_if_fail (!checker || SPELLING_IS_CHECKER (checker));

  if (self->checker == checker)
    return;

  if (self->checker)
    g_signal_handlers_disconnect_by_func (self->checker,
                                          G_CALLBACK (spelling_text_buffer_adapter_checker_notify_language),
                                          self);

  g_set_object (&self->checker, checker);

  if (checker)
    {
      g_signal_connect_object (self->checker,
                               "notify::language",
                               G_CALLBACK (spelling_text_buffer_adapter_checker_notify_language),
                               self,
                               G_CONNECT_SWAPPED);

      if (!(code = spelling_checker_get_language (checker)))
        code = "";
    }

  spelling_engine_invalidate_all (self->engine);

  spelling_text_buffer_adapter_set_action_state (self, "language", g_variant_new_string (code));

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CHECKER]);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LANGUAGE]);
}

/**
 * spelling_text_buffer_adapter_get_buffer:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Gets the underlying buffer for the adapter.
 *
 * Returns: (transfer none) (nullable): a `GtkSourceBuffer`
 */
GtkSourceBuffer *
spelling_text_buffer_adapter_get_buffer (SpellingTextBufferAdapter *self)
{
  g_autoptr(GtkTextBuffer) buffer = NULL;

  g_return_val_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self), NULL);

  buffer = g_weak_ref_get (&self->buffer_wr);

  /* return borrowed instance only */
  return GTK_SOURCE_BUFFER (buffer);
}

/**
 * spelling_text_buffer_adapter_get_language:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Gets the checker language.
 *
 * Returns: (transfer none) (nullable): a language code
 */
const char *
spelling_text_buffer_adapter_get_language (SpellingTextBufferAdapter *self)
{
  g_return_val_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self), NULL);

  return self->checker ? spelling_checker_get_language (self->checker) : NULL;
}

/**
 * spelling_text_buffer_adapter_set_language:
 * @self: a `SpellingTextBufferAdapter`
 * @language: the language to use
 *
 * Sets the language code to use by the checker, such as `en_US`.
 */
void
spelling_text_buffer_adapter_set_language (SpellingTextBufferAdapter *self,
                                           const char                *language)
{
  g_return_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  if (self->checker == NULL && language == NULL)
    return;

  if (self->checker == NULL)
    {
      self->checker = spelling_checker_new (NULL, language);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CHECKER]);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LANGUAGE]);
    }
  else if (g_strcmp0 (language, spelling_text_buffer_adapter_get_language (self)) != 0)
    {
      spelling_checker_set_language (self->checker, language);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LANGUAGE]);
    }

  spelling_text_buffer_adapter_invalidate_all (self);
}

/**
 * spelling_text_buffer_adapter_get_tag:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Gets the tag used for potentially misspelled words.
 *
 * Returns: (transfer none) (nullable): a `GtkTextTag` or %NULL
 */
GtkTextTag *
spelling_text_buffer_adapter_get_tag (SpellingTextBufferAdapter *self)
{
  g_return_val_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self), NULL);

  return self->tag;
}

/**
 * spelling_text_buffer_adapter_get_enabled:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Gets if the spellcheck is enabled.
 *
 * Returns: %TRUE if enabled
 */
gboolean
spelling_text_buffer_adapter_get_enabled (SpellingTextBufferAdapter *self)
{
  g_return_val_if_fail (!self || SPELLING_IS_TEXT_BUFFER_ADAPTER (self), FALSE);

  if (self == NULL)
    return FALSE;

  return self->enabled;
}

/**
 * spelling_text_buffer_adapter_get_menu_model:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Gets the menu model containing corrections
 *
 * Returns: (transfer none): a `GMenuModel`
 */
GMenuModel *
spelling_text_buffer_adapter_get_menu_model (SpellingTextBufferAdapter *self)
{
  g_return_val_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self), NULL);

  if (self->menu == NULL)
    {
      self->menu = spelling_menu_new ();
      self->top_menu = g_menu_new ();
      g_menu_append_section (self->top_menu, NULL, self->menu);
    }

  return G_MENU_MODEL (self->top_menu);
}

static void
spelling_add_action (SpellingTextBufferAdapter *self,
                     GVariant                  *param)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (self->word_under_cursor != NULL);

  if (self->checker != NULL)
    {
      spelling_checker_add_word (self->checker, self->word_under_cursor);
      spelling_text_buffer_adapter_invalidate_all (self);
    }
}

static void
spelling_ignore_action (SpellingTextBufferAdapter *self,
                        GVariant                  *param)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (self->word_under_cursor != NULL);

  if (self->checker != NULL)
    {
      spelling_checker_ignore_word (self->checker, self->word_under_cursor);
      spelling_text_buffer_adapter_invalidate_all (self);
    }
}

static void
spelling_enabled_action (SpellingTextBufferAdapter *self,
                         GVariant                  *param)
{
  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  spelling_text_buffer_adapter_set_enabled (self,
                                            !spelling_text_buffer_adapter_get_enabled (self));
}

static void
spelling_correct_action (SpellingTextBufferAdapter *self,
                         GVariant                  *param)
{
  g_autofree char *slice = NULL;
  g_autoptr(GtkTextBuffer) buffer = NULL;
  const char *word;
  GtkTextIter begin, end;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (g_variant_is_of_type (param, G_VARIANT_TYPE_STRING));

  if (!(buffer = g_weak_ref_get (&self->buffer_wr)))
    return;

  word = g_variant_get_string (param, NULL);

  /* We don't deal with selections (yet?) */
  if (gtk_text_buffer_get_selection_bounds (buffer, &begin, &end))
    return;

  if (!get_word_at_position (self, gtk_text_iter_get_offset (&begin), &begin, &end))
    return;

  slice = gtk_text_iter_get_slice (&begin, &end);

  if (g_strcmp0 (slice, self->word_under_cursor) != 0)
    {
      g_debug ("Words do not match, will not replace.");
      return;
    }

  gtk_text_buffer_begin_user_action (buffer);
  gtk_text_buffer_delete (buffer, &begin, &end);
  gtk_text_buffer_insert (buffer, &begin, word, -1);
  gtk_text_buffer_end_user_action (buffer);
}

static void
spelling_language_action (SpellingTextBufferAdapter *self,
                          GVariant                  *param)
{
  const char *code;

  g_assert (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));
  g_assert (g_variant_is_of_type (param, G_VARIANT_TYPE_STRING));

  code = g_variant_get_string (param, NULL);

  if (self->checker)
    spelling_checker_set_language (self->checker, code);
}

/**
 * spelling_text_buffer_adapter_update_corrections:
 * @self: a `SpellingTextBufferAdapter`
 *
 * Looks at the current cursor position and updates the list of
 * corrections based on the current word.
 *
 * Use this to force an update immediately rather than after the
 * automatic timeout caused by cursor movements.
 */
void
spelling_text_buffer_adapter_update_corrections (SpellingTextBufferAdapter *self)
{
  g_return_if_fail (SPELLING_IS_TEXT_BUFFER_ADAPTER (self));

  if (self->enabled == FALSE)
    return;

  remember_word_under_cursor (self);
}
