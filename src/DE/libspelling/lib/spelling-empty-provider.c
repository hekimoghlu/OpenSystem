/* spelling-empty-provider.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
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

#include "spelling-empty-provider-private.h"
#include "spelling-language.h"

struct _SpellingEmptyProvider
{
  SpellingProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (SpellingEmptyProvider, spelling_empty_provider, SPELLING_TYPE_PROVIDER)

SpellingProvider *
spelling_empty_provider_new (void)
{
  return g_object_new (SPELLING_TYPE_EMPTY_PROVIDER, NULL);
}

static GListModel *
empty_list_languages (SpellingProvider *provider)
{
  return G_LIST_MODEL (g_list_store_new (SPELLING_TYPE_LANGUAGE));
}

static SpellingDictionary *
empty_load_dictionary (SpellingProvider *provider,
                       const char       *language)
{
  return NULL;
}

static gboolean
empty_supports_language (SpellingProvider *provider,
                         const char       *language)
{
  return FALSE;
}

static void
spelling_empty_provider_class_init (SpellingEmptyProviderClass *klass)
{
  SpellingProviderClass *provider_class = SPELLING_PROVIDER_CLASS (klass);

  provider_class->list_languages = empty_list_languages;
  provider_class->load_dictionary = empty_load_dictionary;
  provider_class->supports_language = empty_supports_language;
}

static void
spelling_empty_provider_init (SpellingEmptyProvider *self)
{
}
