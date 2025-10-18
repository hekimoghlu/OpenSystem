/* test-engine.c
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

#include <libspelling.h>

#include "spelling-engine-private.h"

static GString *buffer;
static GtkBitset *mispelled;
static SpellingDictionary *dictionary;
static guint cursor;

typedef struct
{
  GString *str;
  const char *pos;
  guint offset;
  gunichar ch;
} StringIter;

static inline gboolean
string_iter_is_end (StringIter *iter)
{
  return iter->pos >= iter->str->str + iter->str->len;
}

static inline void
string_iter_init (StringIter *iter,
                  GString    *str,
                  guint       offset)
{
  iter->str = str;
  iter->pos = g_utf8_offset_to_pointer (str->str, offset);
  iter->offset = offset;
  iter->ch = g_utf8_get_char (iter->pos);
}

static inline gboolean
string_iter_backward (StringIter *iter)
{
  if (iter->pos <= iter->str->str)
    return FALSE;

  iter->pos = g_utf8_prev_char (iter->pos);
  iter->ch = g_utf8_get_char (iter->pos);
  iter->offset--;

  return TRUE;
}

static inline gboolean
string_iter_forward (StringIter *iter)
{
  if (iter->pos >= iter->str->str + iter->str->len)
    return FALSE;

  iter->pos = g_utf8_next_char (iter->pos);
  iter->ch = g_utf8_get_char (iter->pos);
  iter->offset++;

  return TRUE;
}

static char *
copy_text (gpointer instance,
           guint    position,
           guint    length)
{
  const char *begin = g_utf8_offset_to_pointer (buffer->str, position);
  const char *end = g_utf8_offset_to_pointer (begin, length);

  return g_strndup (begin, end - begin);
}

static void
clear_tag (gpointer instance,
           guint    position,
           guint    length)
{
  gtk_bitset_remove_range (mispelled, position, length);
}

static void
apply_tag (gpointer instance,
           guint    position,
           guint    length)
{
  gtk_bitset_add_range (mispelled, position, length);
}

static gboolean
is_word_char (gunichar ch)
{
  const char *extra_word_chars = spelling_dictionary_get_extra_word_chars (dictionary);

  for (const char *c = extra_word_chars; c && *c; c = g_utf8_next_char (c))
    {
      if (ch == g_utf8_get_char (c))
        return TRUE;
    }

  return g_unichar_isalnum (ch) || ch == '_';
}

static gboolean
backward_word_start (gpointer  instance,
                     guint    *position)
{
  StringIter iter;
  StringIter peek;

  string_iter_init (&iter, buffer, *position);

  /* Move back one char first */
  if (!string_iter_backward (&iter))
    return FALSE;

  /* If we have left a word (into space, etc), walk back until
   * we get to the end of a word.
   */
  if (!is_word_char (iter.ch))
    {
      while (string_iter_backward (&iter))
        {
          if (!is_word_char (iter.ch))
            continue;
          break;
        }

      if (iter.offset == 0 && !is_word_char (iter.ch))
        return FALSE;
    }

  /* Walk backwards saving our last valid position */
  peek = iter;
  while (string_iter_backward (&peek) && is_word_char (peek.ch))
    iter = peek;

  *position = iter.offset;

  return TRUE;
}

static gboolean
forward_word_end (gpointer  instance,
                  guint    *position)
{
  StringIter iter;

  string_iter_init (&iter, buffer, *position);

  /* Move forward one char first */
  if (!string_iter_forward (&iter))
    return FALSE;

  /* If we are no longer on a word character, then walk forward
   * until we reach a word character.
   */
  if (!is_word_char (iter.ch))
    {
      while (string_iter_forward (&iter))
        {
          if (!is_word_char (iter.ch))
            continue;
          break;
        }

      if (string_iter_is_end (&iter))
        return FALSE;
    }

  /* Walk forward saving our last valid position */
  while (string_iter_forward (&iter))
    if (!is_word_char (iter.ch))
      break;

  *position = iter.offset;

  return TRUE;
}

static void
intersect_spellcheck_region (gpointer   instance,
                             GtkBitset *bitset)
{
}

static guint
get_cursor (gpointer instance)
{
  return cursor;
}

static PangoLanguage *
get_language (gpointer instance)
{
  return pango_language_get_default ();
}

static SpellingDictionary *
get_dictionary (gpointer instance)
{
  return dictionary;
}

static gboolean
check_enabled (gpointer instance)
{
  return TRUE;
}

static const SpellingAdapter adapter = {
  .check_enabled = check_enabled,
  .get_cursor = get_cursor,
  .copy_text = copy_text,
  .clear_tag = clear_tag,
  .apply_tag = apply_tag,
  .backward_word_start = backward_word_start,
  .forward_word_end = forward_word_end,
  .intersect_spellcheck_region = intersect_spellcheck_region,
  .get_dictionary = get_dictionary,
  .get_language = get_language,
};

static inline void
assert_string (const char *str)
{
  g_assert_cmpstr (buffer->str, ==, str);
}

static void
insert (SpellingEngine *engine,
        const char     *text,
        guint           position,
        const char     *expected)
{
  const char *ptr;
  guint n_chars;

  if (position == 0)
    ptr = buffer->str;
  else
    ptr = g_utf8_offset_to_pointer (buffer->str, position);

  n_chars = g_utf8_strlen (text, -1);

  spelling_engine_before_insert_text (engine, position, n_chars);
  g_string_insert (buffer, ptr - buffer->str, text);
  gtk_bitset_splice (mispelled, position, 0, n_chars);
  cursor = position + n_chars;
  spelling_engine_after_insert_text (engine, position, n_chars);

  if (expected)
    assert_string (expected);

  spelling_engine_iteration (engine);
}

static void
delete (SpellingEngine *engine,
        guint           position,
        guint           n_chars,
        const char     *expected)
{
  const char *ptr = g_utf8_offset_to_pointer (buffer->str, position);
  const char *endptr = g_utf8_offset_to_pointer (ptr, n_chars);

  cursor = position;
  spelling_engine_before_delete_range (engine, position, n_chars);
  gtk_bitset_splice (mispelled, position, n_chars, 0);
  g_string_erase (buffer, ptr - buffer->str, endptr - ptr);
  spelling_engine_after_delete_range (engine, position);

  if (expected)
    assert_string (expected);

  spelling_engine_iteration (engine);
}

static void
test_engine_basic (void)
{
  g_autoptr(SpellingEngine) engine = NULL;
  g_autoptr(GObject) instance = g_object_new (G_TYPE_OBJECT, NULL);
  SpellingProvider *provider = spelling_provider_get_default ();
  const char *default_code = spelling_provider_get_default_code (provider);

  dictionary = spelling_provider_load_dictionary (provider, default_code);

  buffer = g_string_new (NULL);
  mispelled = gtk_bitset_new_empty ();
  engine = spelling_engine_new (&adapter, instance);

  insert (engine, "2", 0, "2");
  insert (engine, "1", 0, "12");
  insert (engine, "0", 0, "012");

  delete (engine, 2, 1, "01");
  delete (engine, 1, 1, "0");
  delete (engine, 0, 1, "");

  g_string_free (buffer, TRUE);
  gtk_bitset_unref (mispelled);
  g_object_unref (dictionary);
}

int
main (int argc,
      char *argv[])
{
  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Spelling/Engine/basic", test_engine_basic);
  return g_test_run ();
}
