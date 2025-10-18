/* foundry-search-provider.c
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

#include <glib/gi18n-lib.h>

#include "foundry-search-provider-private.h"
#include "foundry-search-request.h"

G_DEFINE_ABSTRACT_TYPE (FoundrySearchProvider, foundry_search_provider, FOUNDRY_TYPE_CONTEXTUAL)

static DexFuture *
foundry_search_provider_real_load (FoundrySearchProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_search_provider_real_unload (FoundrySearchProvider *self)
{
  return dex_future_new_true ();
}

static void
foundry_search_provider_class_init (FoundrySearchProviderClass *klass)
{
  klass->load = foundry_search_provider_real_load;
  klass->unload = foundry_search_provider_real_unload;
}

static void
foundry_search_provider_init (FoundrySearchProvider *self)
{
}

/**
 * foundry_search_provider_load:
 * @self: a #FoundrySearchProvider
 *
 * Returns: (transfer full): a #DexFuture
 */
DexFuture *
foundry_search_provider_load (FoundrySearchProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SEARCH_PROVIDER (self), NULL);

  return FOUNDRY_SEARCH_PROVIDER_GET_CLASS (self)->load (self);
}

/**
 * foundry_search_provider_unload:
 * @self: a #FoundrySearchProvider
 *
 * Returns: (transfer full): a #DexFuture
 */
DexFuture *
foundry_search_provider_unload (FoundrySearchProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SEARCH_PROVIDER (self), NULL);

  return FOUNDRY_SEARCH_PROVIDER_GET_CLASS (self)->unload (self);
}

/**
 * foundry_search_provider_dup_name:
 * @self: a #FoundrySearchProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Flatpak".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_search_provider_dup_name (FoundrySearchProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_SEARCH_PROVIDER (self), NULL);

  if (FOUNDRY_SEARCH_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_SEARCH_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}

/**
 * foundry_search_provider_search:
 * @self: a [class@Foundry.SearchProvider]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.SearchResult] or
 *   rejects with error.
 */
DexFuture *
foundry_search_provider_search (FoundrySearchProvider *self,
                                FoundrySearchRequest  *request)
{
  dex_return_error_if_fail (FOUNDRY_IS_SEARCH_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_SEARCH_REQUEST (request));

  return FOUNDRY_SEARCH_PROVIDER_GET_CLASS (self)->search (self, request);
}
