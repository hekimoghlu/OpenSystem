/* foundry-simple-text-buffer.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "line-reader-private.h"

#include "foundry-context.h"
#include "foundry-simple-text-buffer.h"
#include "foundry-text-edit.h"
#include "foundry-text-iter.h"

struct _FoundrySimpleTextBuffer
{
  GObject         parent_instance;
  FoundryContext *context;
  GString        *contents;
  char           *language_id;
  guint           stamp;
};

enum {
  PROP_0,
  PROP_CONTEXT,
  PROP_LANGUAGE_ID,
  N_PROPS
};

static void text_buffer_iface_init (FoundryTextBufferInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySimpleTextBuffer, foundry_simple_text_buffer, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (FOUNDRY_TYPE_TEXT_BUFFER, text_buffer_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_simple_text_buffer_finalize (GObject *object)
{
  FoundrySimpleTextBuffer *self = (FoundrySimpleTextBuffer *)object;

  g_clear_weak_pointer (&self->context);

  g_clear_pointer (&self->language_id, g_free);

  g_string_free (self->contents, TRUE);
  self->contents = NULL;

  G_OBJECT_CLASS (foundry_simple_text_buffer_parent_class)->finalize (object);
}

static void
foundry_simple_text_buffer_get_property (GObject    *object,
                                         guint       prop_id,
                                         GValue     *value,
                                         GParamSpec *pspec)
{
  FoundrySimpleTextBuffer *self = FOUNDRY_SIMPLE_TEXT_BUFFER (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_value_set_object (value, self->context);
      break;

    case PROP_LANGUAGE_ID:
      g_value_take_string (value, foundry_text_buffer_dup_language_id (FOUNDRY_TEXT_BUFFER (self)));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_simple_text_buffer_set_property (GObject      *object,
                                         guint         prop_id,
                                         const GValue *value,
                                         GParamSpec   *pspec)
{
  FoundrySimpleTextBuffer *self = FOUNDRY_SIMPLE_TEXT_BUFFER (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_set_weak_pointer (&self->context, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_simple_text_buffer_class_init (FoundrySimpleTextBufferClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_simple_text_buffer_finalize;
  object_class->get_property = foundry_simple_text_buffer_get_property;
  object_class->set_property = foundry_simple_text_buffer_set_property;

  properties[PROP_CONTEXT] =
    g_param_spec_object ("context", NULL, NULL,
                         FOUNDRY_TYPE_CONTEXT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LANGUAGE_ID] =
    g_param_spec_string ("language-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_simple_text_buffer_init (FoundrySimpleTextBuffer *self)
{
  self->contents = g_string_new (NULL);
}

/**
 * foundry_simple_text_buffer_new:
 *
 * Returns: (transfer full):
 */
FoundryTextBuffer *
foundry_simple_text_buffer_new (void)
{
  return g_object_new (FOUNDRY_TYPE_SIMPLE_TEXT_BUFFER, NULL);
}

/**
 * foundry_simple_text_buffer_new_for_string:
 *
 * Returns: (transfer full):
 */
FoundryTextBuffer *
foundry_simple_text_buffer_new_for_string (const char *string,
                                           gssize      len)
{
  FoundrySimpleTextBuffer *self;

  self = g_object_new (FOUNDRY_TYPE_SIMPLE_TEXT_BUFFER, NULL);

  if (len < 0)
    len = strlen (string);

  if (string != NULL && string[0] != 0)
    g_string_append_len (self->contents, string, len);

  return FOUNDRY_TEXT_BUFFER (self);
}

static GBytes *
foundry_simple_text_buffer_dup_contents (FoundryTextBuffer *text_buffer)
{
  FoundrySimpleTextBuffer *self = FOUNDRY_SIMPLE_TEXT_BUFFER (text_buffer);
  char *copy;

  copy = g_malloc (self->contents->len + 1);
  memcpy (copy, self->contents->str, self->contents->len);
  copy[self->contents->len] = 0;

  return g_bytes_new_take (copy, self->contents->len);
}

static DexFuture *
foundry_simple_text_buffer_settle (FoundryTextBuffer *text_buffer)
{
  return dex_future_new_true ();
}

static void
order (gsize *a,
       gsize *b)
{
  if (*a > *b)
    {
      gsize tmp = *a;
      *a = *b;
      *b = tmp;
    }
}

static void
foundry_simple_text_buffer_get_offset_at (FoundrySimpleTextBuffer *self,
                                          gsize                   *iter,
                                          guint                    line,
                                          int                      line_offset)
{
  LineReader reader;
  const char *str;
  gsize line_len;

  g_assert (FOUNDRY_IS_SIMPLE_TEXT_BUFFER (self));
  g_assert (iter != NULL);

  *iter = 0;

  line_reader_init (&reader, self->contents->str, self->contents->len);

  while ((str = line_reader_next (&reader, &line_len)))
    {
      if (line == 0)
        {
          gsize n_chars = g_utf8_strlen (str, line_len);

          if (line_offset < 0 || line_offset >= n_chars)
            *iter = str - self->contents->str + line_len;
          else
            *iter = g_utf8_offset_to_pointer (str, line_offset) - self->contents->str;

          return;
        }

      line--;
    }

  *iter = self->contents->len;
}

static gboolean
foundry_simple_text_buffer_apply_edit (FoundryTextBuffer *text_editor,
                                       FoundryTextEdit   *edit)
{
  FoundrySimpleTextBuffer *self = (FoundrySimpleTextBuffer *)text_editor;
  g_autofree char *replacement = NULL;
  guint begin_line;
  guint end_line;
  int begin_line_offset;
  int end_line_offset;
  gsize begin;
  gsize end;

  g_assert (FOUNDRY_IS_SIMPLE_TEXT_BUFFER (self));
  g_assert (FOUNDRY_IS_TEXT_EDIT (edit));

  foundry_text_edit_get_range (edit,
                               &begin_line, &begin_line_offset,
                               &end_line, &end_line_offset);

  foundry_simple_text_buffer_get_offset_at (self, &begin, begin_line, begin_line_offset);
  foundry_simple_text_buffer_get_offset_at (self, &end, end_line, end_line_offset);

  order (&begin, &end);

  g_string_erase (self->contents, begin, end - begin);

  if ((replacement = foundry_text_edit_dup_replacement (edit)))
    g_string_insert (self->contents, begin, replacement);

  self->stamp++;

  foundry_text_buffer_emit_changed (FOUNDRY_TEXT_BUFFER (self));

  return TRUE;
}

typedef union _FoundrySimpleTextIter
{
  FoundryTextIter iter;
  struct {
    FoundryTextBuffer           *buffer;
    const FoundryTextIterVTable *vtable;
    const char                  *ptr;
    guint                        stamp;
    guint                        offset;
    guint                        line;
    guint                        line_offset;
  };
} FoundrySimpleTextIter;

static gboolean
foundry_simple_text_iter_check (const FoundryTextIter *iter)
{
  const FoundrySimpleTextIter *simple = (const FoundrySimpleTextIter *)iter;

  return simple != NULL &&
         simple->buffer != NULL &&
         FOUNDRY_IS_SIMPLE_TEXT_BUFFER (simple->buffer) &&
         FOUNDRY_SIMPLE_TEXT_BUFFER (simple->buffer)->stamp == simple->stamp;
}

static gsize
foundry_simple_text_iter_get_offset (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->offset;
}

static gsize
foundry_simple_text_iter_get_line (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->line;
}

static gsize
foundry_simple_text_iter_get_line_offset (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->line_offset;
}

static gunichar
foundry_simple_text_iter_get_char (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return g_utf8_get_char (simple->ptr);
}

static gboolean
foundry_simple_text_iter_is_start (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->offset == 0;
}

static gboolean
foundry_simple_text_iter_is_end (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->ptr[0] == 0;
}

static gboolean
foundry_simple_text_iter_forward_char (FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  if (simple->ptr[0] == 0)
    return FALSE;

  if (simple->ptr[0] == '\r')
    {
      /* Treat \r\n as one iteration point */
      if (simple->ptr[1] == '\n')
        simple->ptr++;
    }

  if (simple->ptr[0] == '\r' || simple->ptr[0] == '\n')
    {
      simple->line++;
      simple->line_offset = 0;
    }
  else
    {
      simple->line_offset++;
    }

  simple->offset++;
  simple->ptr = g_utf8_next_char (simple->ptr);

  return foundry_simple_text_iter_is_end (iter);
}

static gboolean
foundry_simple_text_iter_backward_char (FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  if (simple->offset == 0)
    return FALSE;

  simple->offset--;
  simple->ptr = g_utf8_prev_char (simple->ptr);

  if (simple->offset == 0)
    {
      simple->line = 0;
      simple->line_offset = 0;
      return TRUE;
    }

  if (simple->line_offset == 0)
    {
      const char *bounds = FOUNDRY_SIMPLE_TEXT_BUFFER (simple->buffer)->contents->str;
      const char *ptr;

      simple->line--;

      if (simple->ptr[0] == '\n' && simple->ptr[-1] == '\r')
        {
          simple->ptr--;

          if (simple->ptr == bounds)
            {
              g_assert (simple->line == 0);
              simple->line_offset = 0;
              return TRUE;
            }
        }

      ptr = simple->ptr;

      do
        {
          ptr = g_utf8_prev_char (ptr);

          if (*ptr == '\n' || *ptr == '\r')
            {
              ptr++;
              break;
            }
        }
      while (ptr > bounds);

      simple->line_offset = g_utf8_strlen (ptr, simple->ptr - ptr);
    }

  return TRUE;
}

static gboolean
foundry_simple_text_iter_starts_line (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->line_offset == 0;
}

static gboolean
foundry_simple_text_iter_ends_line (const FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  return simple->ptr[0] == 0 || simple->ptr[0] == '\n' || simple->ptr[0] == '\r';
}

static void
foundry_simple_text_iter_reset (FoundryTextIter *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  simple->ptr = FOUNDRY_SIMPLE_TEXT_BUFFER (simple->buffer)->contents->str;
  simple->offset = 0;
  simple->line = 0;
  simple->line_offset = 0;
}

static gboolean
foundry_simple_text_iter_move_to_line_and_offset (FoundryTextIter *iter,
                                                  gsize            line,
                                                  gsize            line_offset)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  if (simple->line > line ||
      (simple->line == line && simple->line_offset > line_offset))
    foundry_simple_text_iter_reset (iter);

  while (simple->line < line)
    {
      if (!foundry_simple_text_iter_forward_char (iter))
        break;
    }

  if (simple->line == line)
    {
      while (simple->line_offset < line_offset &&
             !foundry_simple_text_iter_ends_line (iter))
        {
          if (!foundry_simple_text_iter_forward_char (iter))
            break;
        }
    }

  return foundry_simple_text_iter_get_line (iter) == line &&
         foundry_simple_text_iter_get_line_offset (iter) == line_offset;
}

static gboolean
foundry_simple_text_iter_forward_line (FoundryTextIter *iter)
{
  g_return_val_if_fail (foundry_simple_text_iter_check (iter), 0);

  while (!foundry_simple_text_iter_ends_line (iter))
    {
      if (!foundry_simple_text_iter_forward_char (iter))
        return FALSE;
    }

  return foundry_simple_text_iter_forward_char (iter);
}

static FoundryTextIterVTable iter_vtable = {
  .backward_char = foundry_simple_text_iter_backward_char,
  .ends_line = foundry_simple_text_iter_ends_line,
  .forward_char = foundry_simple_text_iter_forward_char,
  .forward_line = foundry_simple_text_iter_forward_line,
  .get_char = foundry_simple_text_iter_get_char,
  .get_line = foundry_simple_text_iter_get_line,
  .get_line_offset = foundry_simple_text_iter_get_line_offset,
  .get_offset = foundry_simple_text_iter_get_offset,
  .is_end = foundry_simple_text_iter_is_end,
  .is_start = foundry_simple_text_iter_is_start,
  .move_to_line_and_offset = foundry_simple_text_iter_move_to_line_and_offset,
  .starts_line = foundry_simple_text_iter_starts_line,
};

static void
foundry_simple_text_buffer_iter_init (FoundryTextBuffer *buffer,
                                      FoundryTextIter   *iter)
{
  FoundrySimpleTextIter *simple = (FoundrySimpleTextIter *)iter;

  memset (simple, 0, sizeof *iter);

  simple->vtable = &iter_vtable;
  simple->buffer = buffer;
  simple->stamp = FOUNDRY_SIMPLE_TEXT_BUFFER (buffer)->stamp;
  simple->ptr = FOUNDRY_SIMPLE_TEXT_BUFFER (buffer)->contents->str;
  simple->offset = 0;
  simple->line = 0;
  simple->line_offset = 0;
}

static gint64
foundry_simple_text_buffer_get_change_count (FoundryTextBuffer *buffer)
{
  return FOUNDRY_SIMPLE_TEXT_BUFFER (buffer)->stamp;
}

static char *
foundry_simple_text_buffer_dup_language_id (FoundryTextBuffer *buffer)
{
  FoundrySimpleTextBuffer *self = FOUNDRY_SIMPLE_TEXT_BUFFER (buffer);

  return g_strdup (self->language_id);
}

static void
text_buffer_iface_init (FoundryTextBufferInterface *iface)
{
  iface->dup_contents = foundry_simple_text_buffer_dup_contents;
  iface->settle = foundry_simple_text_buffer_settle;
  iface->apply_edit = foundry_simple_text_buffer_apply_edit;
  iface->iter_init = foundry_simple_text_buffer_iter_init;
  iface->get_change_count = foundry_simple_text_buffer_get_change_count;
  iface->dup_language_id = foundry_simple_text_buffer_dup_language_id;
}

void
foundry_simple_text_buffer_set_text (FoundrySimpleTextBuffer *self,
                                     const char              *text,
                                     gssize                   text_len)
{
  g_return_if_fail (FOUNDRY_IS_SIMPLE_TEXT_BUFFER (self));
  g_return_if_fail (text != NULL);

  if (text_len < 0)
    text_len = strlen (text);

  g_string_truncate (self->contents, 0);
  g_string_append_len (self->contents, text, text_len);

  self->stamp++;

  foundry_text_buffer_emit_changed (FOUNDRY_TEXT_BUFFER (self));
}

void
foundry_simple_text_buffer_set_language_id (FoundrySimpleTextBuffer *self,
                                            const char              *language_id)
{
  g_return_if_fail (FOUNDRY_IS_SIMPLE_TEXT_BUFFER (self));

  g_set_str (&self->language_id, language_id);
}
