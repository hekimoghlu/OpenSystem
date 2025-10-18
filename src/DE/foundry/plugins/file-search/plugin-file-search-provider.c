/* plugin-file-search-provider.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
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

#include "plugin-file-search-provider.h"
#include "plugin-file-search-service.h"

struct _PluginFileSearchProvider
{
  FoundrySearchProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginFileSearchProvider, plugin_file_search_provider, FOUNDRY_TYPE_SEARCH_PROVIDER)

static DexFuture *
plugin_file_search_provider_search (FoundrySearchProvider *provider,
                                    FoundrySearchRequest  *request)
{
  g_autoptr(PluginFileSearchService) service = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *search_text = NULL;

  dex_return_error_if_fail (PLUGIN_IS_FILE_SEARCH_PROVIDER (provider));
  dex_return_error_if_fail (FOUNDRY_IS_SEARCH_REQUEST (request));

  if (!foundry_search_request_has_category (request, FOUNDRY_SEARCH_CATEGORY_FILES))
    return foundry_future_new_not_supported ();

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));
  service = foundry_context_dup_service_typed (context, PLUGIN_TYPE_FILE_SEARCH_SERVICE);
  search_text = foundry_search_request_dup_search_text (request);

  return plugin_file_search_service_query (service, search_text);
}

static void
plugin_file_search_provider_class_init (PluginFileSearchProviderClass *klass)
{
  FoundrySearchProviderClass *search_provider_class = FOUNDRY_SEARCH_PROVIDER_CLASS (klass);

  search_provider_class->search = plugin_file_search_provider_search;
}

static void
plugin_file_search_provider_init (PluginFileSearchProvider *self)
{
}
