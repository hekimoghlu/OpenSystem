/* spelling-job.c
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

#include <pango/pango.h>

#include "spelling-dictionary-internal.h"
#include "spelling-job-private.h"
#include "spelling-trace.h"

#define GDK_ARRAY_NAME spelling_boundaries
#define GDK_ARRAY_TYPE_NAME SpellingBoundaries
#define GDK_ARRAY_ELEMENT_TYPE SpellingBoundary
#define GDK_ARRAY_BY_VALUE 1
#define GDK_ARRAY_PREALLOC 8
#define GDK_ARRAY_NO_MEMSET
#include "gdkarrayimpl.c"

typedef struct _SpellingFragment
{
  GBytes  *bytes;
  guint    position;
  guint    length;
  gboolean must_discard;
} SpellingFragment;

typedef struct _SpellingMistakes
{
  const SpellingFragment *fragment;
  GArray                 *boundaries;
} SpellingMistakes;

struct _SpellingJob
{
  GObject             parent_instance;
  SpellingDictionary *dictionary;
  PangoLanguage      *language;
  char               *extra_word_chars;
  GArray             *fragments;
  guint               frozen : 1;
};

enum {
  PROP_0,
  PROP_DICTIONARY,
  PROP_LANGUAGE,
  PROP_EXTRA_WORD_CHARS,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (SpellingJob, spelling_job, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
clear_fragment (gpointer data)
{
  SpellingFragment *fragment = data;

  g_clear_pointer (&fragment->bytes, g_bytes_unref);
}

static void
spelling_job_dispose (GObject *object)
{
  SpellingJob *self = (SpellingJob *)object;

  g_clear_object (&self->dictionary);
  g_clear_pointer (&self->fragments, g_array_unref);
  g_clear_pointer (&self->extra_word_chars, g_free);

  self->language = NULL;

  G_OBJECT_CLASS (spelling_job_parent_class)->dispose (object);
}

static void
spelling_job_get_property (GObject    *object,
                           guint       prop_id,
                           GValue     *value,
                           GParamSpec *pspec)
{
  SpellingJob *self = SPELLING_JOB (object);

  switch (prop_id)
    {
    case PROP_DICTIONARY:
      g_value_set_object (value, self->dictionary);
      break;

    case PROP_EXTRA_WORD_CHARS:
      g_value_set_string (value, self->extra_word_chars);
      break;

    case PROP_LANGUAGE:
      g_value_set_pointer (value, self->language);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_job_set_property (GObject      *object,
                           guint         prop_id,
                           const GValue *value,
                           GParamSpec   *pspec)
{
  SpellingJob *self = SPELLING_JOB (object);

  switch (prop_id)
    {
    case PROP_DICTIONARY:
      self->dictionary = g_value_dup_object (value);
      break;

    case PROP_EXTRA_WORD_CHARS:
      self->extra_word_chars = g_value_dup_string (value);
      break;

    case PROP_LANGUAGE:
      self->language = g_value_get_pointer (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_job_class_init (SpellingJobClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = spelling_job_dispose;
  object_class->get_property = spelling_job_get_property;
  object_class->set_property = spelling_job_set_property;

  properties[PROP_DICTIONARY] =
    g_param_spec_object ("dictionary", NULL, NULL,
                         SPELLING_TYPE_DICTIONARY,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_EXTRA_WORD_CHARS] =
    g_param_spec_string ("extra-word-chars", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LANGUAGE] =
    g_param_spec_pointer ("language", NULL, NULL,
                          (G_PARAM_READWRITE |
                           G_PARAM_CONSTRUCT_ONLY |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
spelling_job_init (SpellingJob *self)
{
  self->fragments = g_array_new (FALSE, FALSE, sizeof (SpellingFragment));
  g_array_set_clear_func (self->fragments, clear_fragment);
}

SpellingJob *
spelling_job_new (SpellingDictionary *dictionary,
                  PangoLanguage      *language)
{
  const char *extra_word_chars;

  g_return_val_if_fail (SPELLING_IS_DICTIONARY (dictionary), NULL);
  g_return_val_if_fail (language != NULL, NULL);

  extra_word_chars = spelling_dictionary_get_extra_word_chars (dictionary);

  return g_object_new (SPELLING_TYPE_JOB,
                       "dictionary", dictionary,
                       "extra-word-chars", extra_word_chars,
                       "language", language,
                       NULL);
}

static inline gboolean
is_extra_word_char (const char *extra_word_chars,
                    gunichar    ch)
{
  if (extra_word_chars == NULL)
    return FALSE;

  for (const char *c = extra_word_chars; *c; c = g_utf8_next_char (c))
    {
      if (ch == g_utf8_get_char (c))
        return TRUE;
    }

  return FALSE;
}

static gboolean
find_word_start (const char         **textptr,
                 gsize               *iptr,
                 const PangoLogAttr  *attrs,
                 gsize                attrslen,
                 const char          *extra_word_chars)
{
  while (*iptr < attrslen)
    {
      if (attrs[*iptr].is_word_start)
        return TRUE;

      if (!attrs[*iptr].is_white)
        {
          gunichar ch = g_utf8_get_char (*textptr);

          if (is_extra_word_char (extra_word_chars, ch))
            return TRUE;
        }

      *textptr = g_utf8_next_char (*textptr);
      (*iptr)++;
    }

  return FALSE;
}

static gboolean
find_word_end (const char         **textptr,
               gsize               *iptr,
               const PangoLogAttr  *attrs,
               gsize                attrslen,
               const char          *extra_word_chars)
{
  while (*iptr < attrslen)
    {
      if (attrs[*iptr].is_word_end)
        {
          gboolean skipped = FALSE;

          /* We're at a word boundary, but we might have an extra word
           * char here. If so, skip past the word char.
           */
          while (*iptr < attrslen &&
                 !attrs[*iptr].is_white &&
                 is_extra_word_char (extra_word_chars, g_utf8_get_char (*textptr)))
            {
              skipped = TRUE;
              *textptr = g_utf8_next_char (*textptr);
              (*iptr)++;
            }

          /* If we landed on a word start we must continue as it might
           * be something like `words's` where `'` is the extra word char
           * but `s` is not in extra_word_chars.
           */
          if (skipped &&
              *iptr < attrslen &&
              attrs[*iptr].is_word_start)
            (void)find_word_end (textptr, iptr, attrs, attrslen, extra_word_chars);

          return TRUE;
        }

      *textptr = g_utf8_next_char (*textptr);
      (*iptr)++;
    }

  return FALSE;
}

static void
clear_mistakes (gpointer data)
{
  SpellingMistakes *mistakes = data;

  mistakes->fragment = NULL;
  g_clear_pointer (&mistakes->boundaries, g_array_unref);
}

static void
spelling_job_check (GTask        *task,
                    gpointer      source_object,
                    gpointer      task_data,
                    GCancellable *cancellable)
{
  SpellingJob *self = source_object;
  SpellingBoundaries boundaries;
  g_autoptr(GArray) result = NULL;

  g_assert (G_IS_TASK (task));
  g_assert (SPELLING_IS_JOB (self));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  spelling_boundaries_init (&boundaries);

  result = g_array_new (FALSE, FALSE, sizeof (SpellingMistakes));
  g_array_set_clear_func (result, clear_mistakes);

  SPELLING_PROFILER_LOG ("Checking %u fragments", self->fragments->len);

  for (guint f = 0; f < self->fragments->len; f++)
    {
      const SpellingFragment *fragment = &g_array_index (self->fragments, SpellingFragment, f);
      G_GNUC_UNUSED g_autofree char *message = NULL;
      g_autoptr(GtkBitset) bitset = NULL;
      g_autofree PangoLogAttr *attrs = NULL;
      SpellingMistakes mistakes;
      const char *text;
      const char *p;
      GtkBitsetIter iter;
      gsize textlen;
      gsize attrslen;
      gsize i;
      guint pos;

      SPELLING_PROFILER_BEGIN_MARK;

      spelling_boundaries_clear (&boundaries);

      mistakes.fragment = fragment;
      mistakes.boundaries = NULL;

      text = g_bytes_get_data (fragment->bytes, &textlen);
      attrslen = g_utf8_strlen (text, textlen) + 1;
      attrs = g_new0 (PangoLogAttr, attrslen);

      g_assert (textlen <= G_MAXINT);
      g_assert (attrslen <= G_MAXINT);

      pango_get_log_attrs (text, (int)textlen, -1, self->language, attrs, attrslen);

      p = text;
      i = 0;

      for (gsize count = 0; TRUE; count++)
        {
          SpellingBoundary boundary;
          const char *before = p;

          /* Occasionally check to break out of large runs */
          if ((count & 0xFF) == 0 && g_atomic_int_get (&fragment->must_discard))
            break;

          /* Find next word start */
          if (!find_word_start (&p, &i, attrs, attrslen-1, self->extra_word_chars))
            break;

          boundary.byte_offset = p - text;
          boundary.offset = i;

          /* Ensure we've moved at least one character as find_word_end() may stop
           * on the current character it is on.
           */
          if (p == before)
            {
              p = g_utf8_next_char (p);
              i++;
            }

          if (!find_word_end (&p, &i, attrs, attrslen-1, self->extra_word_chars))
            break;

          boundary.length = i - boundary.offset;
          boundary.byte_length = p - text - boundary.byte_offset;

          if (boundary.byte_length > 0)
            spelling_boundaries_append (&boundaries, &boundary);
        }

      if (g_atomic_int_get (&fragment->must_discard))
        continue;

      bitset = _spelling_dictionary_check_words (self->dictionary,
                                                 text,
                                                 spelling_boundaries_index (&boundaries, 0),
                                                 spelling_boundaries_get_size (&boundaries));

      if (gtk_bitset_iter_init_first (&iter, bitset, &pos))
        {
          mistakes.boundaries = g_array_new (FALSE, FALSE, sizeof (SpellingBoundary));

          do
            {
              const SpellingBoundary *b = spelling_boundaries_index (&boundaries, pos);
              g_array_append_vals (mistakes.boundaries, b, 1);
            }
          while (gtk_bitset_iter_next (&iter, &pos));

          g_array_append_val (result, mistakes);
        }

      if G_UNLIKELY (SPELLING_PROFILER_ACTIVE)
        message = g_strdup_printf ("%u chars, %u bytes, %u mistakes",
                                   (guint)attrslen,
                                   (guint)textlen,
                                   mistakes.boundaries ? mistakes.boundaries->len : 0);

      SPELLING_PROFILER_END_MARK ("Check", message);
    }

  spelling_boundaries_clear (&boundaries);

  g_task_return_pointer (task,
                         g_steal_pointer (&result),
                         (GDestroyNotify)g_array_unref);
}

void
spelling_job_run (SpellingJob         *self,
                  GAsyncReadyCallback  callback,
                  gpointer             user_data)
{
  g_autoptr(GTask) task = NULL;

  g_return_if_fail (SPELLING_IS_JOB (self));

  self->frozen = TRUE;

  task = g_task_new (self, NULL, callback, user_data);
  g_task_set_source_tag (task, spelling_job_run);
  g_task_run_in_thread (task, spelling_job_check);
}

void
spelling_job_run_finish (SpellingJob       *self,
                         GAsyncResult      *result,
                         SpellingBoundary **fragments,
                         guint             *n_fragments,
                         SpellingMistake  **mistakes,
                         guint             *n_mistakes)
{
  g_autoptr(GArray) ar = NULL;

  g_return_if_fail (SPELLING_IS_JOB (self));
  g_return_if_fail (G_IS_TASK (result));
  g_return_if_fail (n_fragments != NULL || fragments == NULL);
  g_return_if_fail (mistakes != NULL);
  g_return_if_fail (n_mistakes != NULL);

  *n_mistakes = 0;
  *mistakes = NULL;

  if (n_fragments != NULL)
    *n_fragments = 0;

  if (fragments != NULL)
    *fragments = NULL;

  if (n_fragments != NULL)
    {
      for (guint i = 0; i < self->fragments->len; i++)
        {
          const SpellingFragment *fragment = &g_array_index (self->fragments, SpellingFragment, i);

          if (fragment->must_discard)
            continue;

          (*n_fragments)++;
        }
    }

  if (fragments != NULL)
    {
      guint pos = 0;

      *fragments = g_new0 (SpellingBoundary, *n_fragments);

      for (guint i = 0; i < self->fragments->len; i++)
        {
          const SpellingFragment *fragment = &g_array_index (self->fragments, SpellingFragment, i);

          if (fragment->must_discard)
            continue;

          (*fragments)[pos].offset = fragment->position;
          (*fragments)[pos].length = fragment->length;

          pos++;
        }
    }

  ar = g_task_propagate_pointer (G_TASK (result), NULL);

  if (ar != NULL)
    {
      guint pos = 0;

      for (guint i = 0; i < ar->len; i++)
        {
          const SpellingMistakes *m = &g_array_index (ar, SpellingMistakes, i);

          if (m->fragment->must_discard)
            continue;

          *n_mistakes += m->boundaries->len;
        }

      if (*n_mistakes == 0)
        return;

      *mistakes = g_new0 (SpellingMistake, *n_mistakes);

      for (guint i = 0; i < ar->len; i++)
        {
          const SpellingMistakes *m = &g_array_index (ar, SpellingMistakes, i);

          if (m->fragment->must_discard)
            continue;

          for (guint j = 0; j < m->boundaries->len; j++)
            {
              const SpellingBoundary *boundary = &g_array_index (m->boundaries, SpellingBoundary, j);

              (*mistakes)[pos].offset = m->fragment->position + boundary->offset;
              (*mistakes)[pos].length = boundary->length;

              pos++;
            }
        }

      g_assert (pos == *n_mistakes);
    }
}

void
spelling_job_run_sync (SpellingJob       *self,
                       SpellingBoundary **fragments,
                       guint             *n_fragments,
                       SpellingMistake  **mistakes,
                       guint             *n_mistakes)
{
  g_autoptr(GTask) task = NULL;

  g_return_if_fail (SPELLING_IS_JOB (self));
  g_return_if_fail (n_fragments != NULL || fragments == NULL);
  g_return_if_fail (mistakes != NULL);
  g_return_if_fail (n_mistakes != NULL);

  self->frozen = TRUE;

  task = g_task_new (self, NULL, NULL, NULL);
  g_task_set_source_tag (task, spelling_job_run);
  spelling_job_check (task, self, NULL, NULL);

  spelling_job_run_finish (self, G_ASYNC_RESULT (task), fragments, n_fragments, mistakes, n_mistakes);
}

void
spelling_job_add_fragment (SpellingJob *self,
                           GBytes      *bytes,
                           guint        position,
                           guint        length)
{
  SpellingFragment fragment = {0};

  g_return_if_fail (SPELLING_IS_JOB (self));
  g_return_if_fail (bytes != NULL);
  g_return_if_fail (self->frozen == FALSE);

  fragment.bytes = g_bytes_ref (bytes);
  fragment.position = position;
  fragment.length = length;
  fragment.must_discard = FALSE;

  g_array_append_val (self->fragments, fragment);
}

void
spelling_job_notify_insert (SpellingJob *self,
                            guint        position,
                            guint        length)
{
  g_return_if_fail (SPELLING_IS_JOB (self));

  for (guint i = 0; i < self->fragments->len; i++)
    {
      SpellingFragment *fragment = &g_array_index (self->fragments, SpellingFragment, i);

      if (fragment->must_discard)
        continue;

      /* Inserts after w/ at least 1 position after fragment */
      if (position > fragment->position + fragment->length)
        continue;

      /* Inserts before w/ at least 1 position before fragment */
      if (position < fragment->position)
        {
          fragment->position += length;
          continue;
        }

      g_atomic_int_set (&fragment->must_discard, TRUE);
    }
}

void
spelling_job_notify_delete (SpellingJob *self,
                            guint        position,
                            guint        length)
{
  g_return_if_fail (SPELLING_IS_JOB (self));

  for (guint i = 0; i < self->fragments->len; i++)
    {
      SpellingFragment *fragment = &g_array_index (self->fragments, SpellingFragment, i);

      if (fragment->must_discard)
        continue;

      /* Deletes after w/ at least 1 position after fragment */
      if (position > fragment->position + fragment->length)
        continue;

      /* If we had the ability to look back at text to see if a boundary
       * character was before the cursor here, we could potentially avoid
       * bailing. But that is more effort than it's worth when we can just
       * recheck things.
       */

      /* Deletes before w/ at least 1 position before fragment */
      if (position + length < fragment->position)
        {
          fragment->position -= length;
          continue;
        }

      g_atomic_int_set (&fragment->must_discard, TRUE);
    }
}

void
spelling_job_invalidate (SpellingJob *self,
                         guint        position,
                         guint        length)
{
  g_return_if_fail (SPELLING_IS_JOB (self));

  for (guint i = 0; i < self->fragments->len; i++)
    {
      SpellingFragment *fragment = &g_array_index (self->fragments, SpellingFragment, i);

      if (fragment->must_discard)
        continue;

      if (position > fragment->position + fragment->length)
        continue;

      if (position + length < fragment->position)
        continue;

      g_atomic_int_set (&fragment->must_discard, TRUE);
    }
}
