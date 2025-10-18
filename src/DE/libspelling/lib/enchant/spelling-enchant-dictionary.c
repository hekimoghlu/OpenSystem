/* spelling-enchant-dictionary.c
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

#include <pango/pango.h>
#include <enchant.h>

#include "spelling-enchant-dictionary.h"

#define MAX_RESULTS 10

struct _SpellingEnchantDictionary
{
  SpellingDictionary parent_instance;
  PangoLanguage *language;
  EnchantDict *native;
  char *extra_word_chars;
};

G_DEFINE_FINAL_TYPE (SpellingEnchantDictionary, spelling_enchant_dictionary, SPELLING_TYPE_DICTIONARY)

enum {
  PROP_0,
  PROP_NATIVE,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

/**
 * spelling_enchant_dictionary_new:
 *
 * Create a new `SpellingEnchantDictionary`.
 *
 * Returns: (transfer full): a newly created `SpellingEnchantDictionary`
 */
SpellingDictionary *
spelling_enchant_dictionary_new (const char *code,
                                 gpointer    native)
{
  return g_object_new (SPELLING_TYPE_ENCHANT_DICTIONARY,
                       "code", code,
                       "native", native,
                       NULL);
}

static inline gboolean
word_is_number (const char *word,
                gsize       word_len)
{
  g_assert (word_len > 0);

  for (gsize i = 0; i < word_len; i++)
    {
      if (word[i] < '0' || word[i] > '9')
        return FALSE;
    }

  return TRUE;
}

static gboolean
spelling_enchant_dictionary_contains_word (SpellingDictionary *dictionary,
                                           const char         *word,
                                           gssize              word_len)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)dictionary;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));
  g_assert (word != NULL);
  g_assert (word_len >= 0);

  if (word_is_number (word, word_len))
    return TRUE;

  return enchant_dict_check (self->native, word, word_len) == 0;
}

static char **
strv_copy_n (const char * const *strv,
             gsize               n)
{
  char **copy = g_new (char *, n + 1);

  for (gsize i = 0; i < n; i++)
    copy[i] = g_strdup (strv[i]);

  copy[n] = NULL;

  return copy;
}

static char **
spelling_enchant_dictionary_list_corrections (SpellingDictionary *dictionary,
                                              const char         *word,
                                              gssize              word_len)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)dictionary;
  size_t count = 0;
  char **tmp;
  char **ret = NULL;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));
  g_assert (word != NULL);
  g_assert (word_len > 0);

  if ((tmp = enchant_dict_suggest (self->native, word, word_len, &count)) && count > 0)
    {
      if (g_strv_length (tmp) <= MAX_RESULTS)
        ret = g_strdupv (tmp);
      else
        ret = strv_copy_n ((const char * const *)tmp, MAX_RESULTS);

      enchant_dict_free_string_list (self->native, tmp);
    }

  return g_steal_pointer (&ret);
}

static char **
spelling_enchant_dictionary_split (SpellingEnchantDictionary *self,
                                   const char                *words)
{
  PangoLogAttr *attrs;
  GArray *ar;
  gsize n_chars;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));

  if (words == NULL || self->language == NULL)
    return NULL;

  /* We don't care about splitting obnoxious stuff */
  if ((n_chars = g_utf8_strlen (words, -1)) > 1024)
    return NULL;

  attrs = g_newa (PangoLogAttr, n_chars + 1);
  pango_get_log_attrs (words, -1, -1, self->language, attrs, n_chars + 1);

  ar = g_array_new (TRUE, FALSE, sizeof (char*));

  for (gsize i = 0; i < n_chars + 1; i++)
    {
      if (attrs[i].is_word_start)
        {
          for (gsize j = i + 1; j < n_chars + 1; j++)
            {
              if (attrs[j].is_word_end)
                {
                  char *substr = g_utf8_substring (words, i, j);
                  g_array_append_val (ar, substr);
                  i = j;
                  break;
                }
            }
        }
    }

  return (char **)(gpointer)g_array_free (ar, FALSE);
}

static void
spelling_enchant_dictionary_add_all_to_session (SpellingEnchantDictionary *self,
                                                const char * const        *words)
{
  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));

  if (words == NULL || words[0] == NULL)
    return;

  for (guint i = 0; words[i]; i++)
    enchant_dict_add_to_session (self->native, words[i], -1);
}

static void
spelling_enchant_dictionary_add_word (SpellingDictionary *dictionary,
                                      const char         *word)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)dictionary;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));
  g_assert (word != NULL);

  enchant_dict_add (self->native, word, -1);
}

static void
spelling_enchant_dictionary_ignore_word (SpellingDictionary *dictionary,
                                         const char         *word)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)dictionary;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));
  g_assert (word != NULL);

  enchant_dict_add_to_session (self->native, word, -1);
}

static const char *
spelling_enchant_dictionary_get_extra_word_chars (SpellingDictionary *dictionary)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)dictionary;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));

  return self->extra_word_chars;
}

static void
spelling_enchant_dictionary_constructed (GObject *object)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)object;
  g_auto(GStrv) split = NULL;
  const char *extra_word_chars;
  const char *code;

  g_assert (SPELLING_IS_ENCHANT_DICTIONARY (self));

  G_OBJECT_CLASS (spelling_enchant_dictionary_parent_class)->constructed (object);

  code = spelling_dictionary_get_code (SPELLING_DICTIONARY (self));
  self->language = pango_language_from_string (code);

  if ((split = spelling_enchant_dictionary_split (self, g_get_real_name ())))
    spelling_enchant_dictionary_add_all_to_session (self, (const char * const *)split);

  if ((extra_word_chars = enchant_dict_get_extra_word_characters (self->native)))
    {
      const char *end_pos = NULL;

      /* Sometimes we get invalid UTF-8 from enchant, so handle that directly.
       * In particular, the data seems corrupted from Fedora.
       */
      if (g_utf8_validate (extra_word_chars, -1, &end_pos))
        self->extra_word_chars = g_strdup (extra_word_chars);
      else
        self->extra_word_chars = g_strndup (extra_word_chars, end_pos - extra_word_chars);
    }
}

static void
spelling_enchant_dictionary_finalize (GObject *object)
{
  SpellingEnchantDictionary *self = (SpellingEnchantDictionary *)object;

  /* Owned by provider */
  self->native = NULL;

  /* Global, no need to free */
  self->language = NULL;

  g_clear_pointer (&self->extra_word_chars, g_free);

  G_OBJECT_CLASS (spelling_enchant_dictionary_parent_class)->finalize (object);
}

static void
spelling_enchant_dictionary_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  SpellingEnchantDictionary *self = SPELLING_ENCHANT_DICTIONARY (object);

  switch (prop_id)
    {
    case PROP_NATIVE:
      g_value_set_pointer (value, self->native);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_enchant_dictionary_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  SpellingEnchantDictionary *self = SPELLING_ENCHANT_DICTIONARY (object);

  switch (prop_id)
    {
    case PROP_NATIVE:
      self->native = g_value_get_pointer (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_enchant_dictionary_class_init (SpellingEnchantDictionaryClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  SpellingDictionaryClass *dictionary_class = SPELLING_DICTIONARY_CLASS (klass);

  object_class->constructed = spelling_enchant_dictionary_constructed;
  object_class->finalize = spelling_enchant_dictionary_finalize;
  object_class->get_property = spelling_enchant_dictionary_get_property;
  object_class->set_property = spelling_enchant_dictionary_set_property;

  dictionary_class->contains_word = spelling_enchant_dictionary_contains_word;
  dictionary_class->list_corrections = spelling_enchant_dictionary_list_corrections;
  dictionary_class->add_word = spelling_enchant_dictionary_add_word;
  dictionary_class->ignore_word = spelling_enchant_dictionary_ignore_word;
  dictionary_class->get_extra_word_chars = spelling_enchant_dictionary_get_extra_word_chars;

  properties[PROP_NATIVE] =
    g_param_spec_pointer ("native",
                          "Native",
                          "The native enchant dictionary",
                          (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
spelling_enchant_dictionary_init (SpellingEnchantDictionary *self)
{
}

gpointer
spelling_enchant_dictionary_get_native (SpellingEnchantDictionary *self)
{
  g_return_val_if_fail (SPELLING_IS_ENCHANT_DICTIONARY (self), NULL);

  return self->native;
}

