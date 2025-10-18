/* spelling-dictionary.c
 *
 * Copyright 2021-2023 Christian Hergert <chergert@redhat.com>
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

#include <string.h>

#include "spelling-dictionary-internal.h"

/**
 * SpellingDictionary:
 *
 * Abstract base class for spellchecking dictionaries.
 */

G_DEFINE_ABSTRACT_TYPE (SpellingDictionary, spelling_dictionary, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_CODE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
spelling_dictionary_real_lock (SpellingDictionary *self)
{
  g_mutex_lock (&self->mutex);
}

static void
spelling_dictionary_real_unlock (SpellingDictionary *self)
{
  g_mutex_unlock (&self->mutex);
}

static inline void
spelling_dictionary_lock (SpellingDictionary *self)
{
  SPELLING_DICTIONARY_GET_CLASS (self)->lock (self);
}

static inline void
spelling_dictionary_unlock (SpellingDictionary *self)
{
  SPELLING_DICTIONARY_GET_CLASS (self)->unlock (self);
}

static void
spelling_dictionary_finalize (GObject *object)
{
  SpellingDictionary *self = (SpellingDictionary *)object;

  self->code = NULL;

  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (spelling_dictionary_parent_class)->finalize (object);
}

static void
spelling_dictionary_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  SpellingDictionary *self = SPELLING_DICTIONARY (object);

  switch (prop_id)
    {
    case PROP_CODE:
      g_value_set_string (value, spelling_dictionary_get_code (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_dictionary_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  SpellingDictionary *self = SPELLING_DICTIONARY (object);

  switch (prop_id)
    {
    case PROP_CODE:
      self->code = g_intern_string (g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_dictionary_class_init (SpellingDictionaryClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = spelling_dictionary_finalize;
  object_class->get_property = spelling_dictionary_get_property;
  object_class->set_property = spelling_dictionary_set_property;

  klass->lock = spelling_dictionary_real_lock;
  klass->unlock = spelling_dictionary_real_unlock;

  /**
   * SpellingDictionary:code:
   *
   * The language code, for example `en_US`.
   */
  properties[PROP_CODE] =
    g_param_spec_string ("code", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
spelling_dictionary_init (SpellingDictionary *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * spelling_dictionary_get_code:
 * @self: a `SpellingDictionary`
 *
 * Gets the language code of the dictionary, or %NULL if undefined.
 *
 * Returns: (transfer none) (nullable): the language code of the dictionary
 */
const char *
spelling_dictionary_get_code (SpellingDictionary *self)
{
  g_return_val_if_fail (SPELLING_IS_DICTIONARY (self), NULL);

  return self->code;
}

/**
 * spelling_dictionary_contains_word:
 * @self: a `SpellingDictionary`
 * @word: a word to be checked
 * @word_len: length of the word, in bytes
 *
 * Checks if the dictionary contains @word.
 *
 * Returns: %TRUE if the dictionary contains the word
 */
gboolean
spelling_dictionary_contains_word (SpellingDictionary *self,
                                   const char         *word,
                                   gssize              word_len)
{
  gboolean ret;

  g_return_val_if_fail (SPELLING_IS_DICTIONARY (self), FALSE);
  g_return_val_if_fail (word != NULL, FALSE);

  if (word_len < 0)
    word_len = strlen (word);

  spelling_dictionary_lock (self);
  ret = SPELLING_DICTIONARY_GET_CLASS (self)->contains_word (self, word, word_len);
  spelling_dictionary_unlock (self);

  return ret;
}

/**
 * spelling_dictionary_list_corrections:
 * @self: a `SpellingDictionary`
 * @word: a word to be checked
 * @word_len: the length of @word, or -1 if @word is zero-terminated
 *
 * Retrieves a list of possible corrections for @word.
 *
 * Returns: (nullable) (transfer full) (array zero-terminated=1) (type utf8):
 *   A list of possible corrections, or %NULL.
 */
char **
spelling_dictionary_list_corrections (SpellingDictionary *self,
                                      const char         *word,
                                      gssize              word_len)
{
  char **ret;

  g_return_val_if_fail (SPELLING_IS_DICTIONARY (self), NULL);
  g_return_val_if_fail (word != NULL, NULL);
  g_return_val_if_fail (word != NULL || word_len == 0, NULL);

  if (word_len < 0)
    word_len = strlen (word);

  if (word_len == 0)
    return NULL;

  spelling_dictionary_lock (self);
  ret = SPELLING_DICTIONARY_GET_CLASS (self)->list_corrections (self, word, word_len);
  spelling_dictionary_unlock (self);

  return ret;
}

/**
 * spelling_dictionary_add_word:
 * @self: a `SpellingDictionary`
 * @word: a word to be added
 *
 * Adds @word to the dictionary.
 */
void
spelling_dictionary_add_word (SpellingDictionary *self,
                              const char         *word)
{
  g_return_if_fail (SPELLING_IS_DICTIONARY (self));
  g_return_if_fail (word != NULL);

  if (SPELLING_DICTIONARY_GET_CLASS (self)->add_word)
    {
      spelling_dictionary_lock (self);
      SPELLING_DICTIONARY_GET_CLASS (self)->add_word (self, word);
      spelling_dictionary_unlock (self);
    }
}

/**
 * spelling_dictionary_ignore_word:
 * @self: a `SpellingDictionary`
 * @word: a word to be ignored
 *
 * Requests the dictionary to ignore @word.
 */
void
spelling_dictionary_ignore_word (SpellingDictionary *self,
                                 const char         *word)
{
  g_return_if_fail (SPELLING_IS_DICTIONARY (self));
  g_return_if_fail (word != NULL);

  if (SPELLING_DICTIONARY_GET_CLASS (self)->ignore_word)
    {
      spelling_dictionary_lock (self);
      SPELLING_DICTIONARY_GET_CLASS (self)->ignore_word (self, word);
      spelling_dictionary_unlock (self);
    }
}

/**
 * spelling_dictionary_get_extra_word_chars:
 * @self: a `SpellingDictionary`
 *
 * Gets the extra word characters of the dictionary.
 *
 * Returns: (transfer none): extra word characters
 */
const char *
spelling_dictionary_get_extra_word_chars (SpellingDictionary *self)
{
  const char *ret = "";

  g_return_val_if_fail (SPELLING_IS_DICTIONARY (self), NULL);

  if (SPELLING_DICTIONARY_GET_CLASS (self)->get_extra_word_chars)
    {
      spelling_dictionary_lock (self);
      ret = SPELLING_DICTIONARY_GET_CLASS (self)->get_extra_word_chars (self);
      spelling_dictionary_unlock (self);
    }

  return ret;
}

GtkBitset *
_spelling_dictionary_check_words (SpellingDictionary     *self,
                                  const char             *text,
                                  const SpellingBoundary *positions,
                                  guint                   n_positions)
{
  gboolean (*contains_word) (SpellingDictionary *, const char *, gssize);
  GtkBitset *bitset;

  g_return_val_if_fail (SPELLING_IS_DICTIONARY (self), NULL);
  g_return_val_if_fail (text != NULL, NULL);

  bitset = gtk_bitset_new_empty ();

  if (n_positions == 0)
    return bitset;

  contains_word = SPELLING_DICTIONARY_GET_CLASS (self)->contains_word;

  spelling_dictionary_lock (self);
  for (guint i = 0; i < n_positions; i++)
    {
      const char *word = &text[positions[i].byte_offset];
      guint wordlen = positions[i].byte_length;

      if (!(*contains_word) (self, word, wordlen))
        gtk_bitset_add (bitset, i);
    }
  spelling_dictionary_unlock (self);

  return bitset;
}
