/* spelling-enchant-provider.c
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

#include <enchant.h>
#include <locale.h>
#include <unicode/uloc.h>

#include <gio/gio.h>

#include "spelling-language-private.h"

#include "spelling-enchant-dictionary.h"
#include "spelling-enchant-provider.h"

struct _SpellingEnchantProvider
{
  SpellingProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (SpellingEnchantProvider, spelling_enchant_provider, SPELLING_TYPE_PROVIDER)

static GHashTable *dictionaries;

static EnchantBroker *
get_broker (void)
{
  static EnchantBroker *broker;

  if (broker == NULL)
    broker = enchant_broker_init ();

  return broker;
}

static char *
_icu_uchar_to_char (const UChar *input,
                    gsize        max_input_len)
{
  GString *str;

  g_assert (input != NULL);
  g_assert (max_input_len > 0);

  if (input[0] == 0)
    return NULL;

  str = g_string_new (NULL);

  for (gsize i = 0; i < max_input_len; i++)
    {
      if (input[i] == 0)
        break;

      g_string_append_unichar (str, input[i]);
    }

  return g_string_free (str, FALSE);
}

static char *
get_display_name (const char *code)
{
  const char * const *names = g_get_language_names ();

  for (guint i = 0; names[i]; i++)
    {
      UChar ret[256];
      UErrorCode status = U_ZERO_ERROR;
      uloc_getDisplayName (code, names[i], ret, G_N_ELEMENTS (ret), &status);
      if (U_SUCCESS (status))
        return _icu_uchar_to_char (ret, G_N_ELEMENTS (ret));
    }

  return NULL;
}

static char *
get_display_language (const char *code)
{
  const char * const *names = g_get_language_names ();

  for (guint i = 0; names[i]; i++)
    {
      UChar ret[256];
      UErrorCode status = U_ZERO_ERROR;
      uloc_getDisplayLanguage (code, names[i], ret, G_N_ELEMENTS (ret), &status);
      if (U_SUCCESS (status))
        return _icu_uchar_to_char (ret, G_N_ELEMENTS (ret));
    }

  return NULL;
}

/**
 * spelling_enchant_provider_new:
 *
 * Create a new `SpellingEnchantProvider`.
 *
 * Returns: (transfer full): a newly created `SpellingEnchantProvider`
 */
SpellingProvider *
spelling_enchant_provider_new (void)
{
  return g_object_new (SPELLING_TYPE_ENCHANT_PROVIDER,
                       "display-name", "Enchant",
                       NULL);
}

static gboolean
spelling_enchant_provider_supports_language (SpellingProvider *provider,
                                             const char       *language)
{
  g_assert (SPELLING_IS_ENCHANT_PROVIDER (provider));
  g_assert (language != NULL);

  return enchant_broker_dict_exists (get_broker (), language);
}

static void
list_languages_cb (const char * const lang_tag,
                   const char * const provider_name,
                   const char * const provider_desc,
                   const char * const provider_file,
                   gpointer           user_data)
{
  GListStore *store = user_data;
  char *name = get_display_name (lang_tag);
  char *group = get_display_language (lang_tag);

  if (name != NULL)
    {
      g_autoptr(SpellingLanguage) language = spelling_language_new (name, lang_tag, group);

      g_list_store_append (store, language);
    }

  g_free (name);
  g_free (group);
}

static GListModel *
spelling_enchant_provider_list_languages (SpellingProvider *provider)
{
  EnchantBroker *broker = get_broker ();
  GListStore *store = g_list_store_new (SPELLING_TYPE_LANGUAGE);
  enchant_broker_list_dicts (broker, list_languages_cb, store);
  return G_LIST_MODEL (store);
}

static SpellingDictionary *
spelling_enchant_provider_load_dictionary (SpellingProvider *provider,
                                           const char       *language)
{
  SpellingDictionary *ret;

  g_assert (SPELLING_IS_ENCHANT_PROVIDER (provider));
  g_assert (language != NULL);

  if (dictionaries == NULL)
    dictionaries = g_hash_table_new_full (g_str_hash, g_str_equal, NULL, g_object_unref);

  if (!(ret = g_hash_table_lookup (dictionaries, language)))
    {
      EnchantDict *dict = enchant_broker_request_dict (get_broker (), language);

      if (dict == NULL)
        return NULL;

      ret = spelling_enchant_dictionary_new (language, dict);
      g_hash_table_insert (dictionaries, (char *)g_intern_string (language), ret);
    }

  return ret ? g_object_ref (ret) : NULL;
}

static void
spelling_enchant_provider_class_init (SpellingEnchantProviderClass *klass)
{
  SpellingProviderClass *spell_provider_class = SPELLING_PROVIDER_CLASS (klass);

  spell_provider_class->supports_language = spelling_enchant_provider_supports_language;
  spell_provider_class->list_languages = spelling_enchant_provider_list_languages;
  spell_provider_class->load_dictionary= spelling_enchant_provider_load_dictionary;
}

static void
spelling_enchant_provider_init (SpellingEnchantProvider *self)
{
}
