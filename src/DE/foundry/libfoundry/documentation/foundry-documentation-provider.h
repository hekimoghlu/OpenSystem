/* foundry-documentation-provider.h
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

#pragma once

#include <libpeas.h>

#include "foundry-config.h"
#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DOCUMENTATION_PROVIDER (foundry_documentation_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDocumentationProvider, foundry_documentation_provider, FOUNDRY, DOCUMENTATION_PROVIDER, FoundryContextual)

struct _FoundryDocumentationProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture  *(*load)          (FoundryDocumentationProvider *self);
  DexFuture  *(*unload)        (FoundryDocumentationProvider *self);
  GListModel *(*list_roots)    (FoundryDocumentationProvider *self);
  DexFuture  *(*index)         (FoundryDocumentationProvider *self,
                                GListModel                   *roots);
  DexFuture  *(*query)         (FoundryDocumentationProvider *self,
                                FoundryDocumentationQuery    *query,
                                FoundryDocumentationMatches  *matches);
  DexFuture  *(*list_children) (FoundryDocumentationProvider *self,
                                FoundryDocumentation         *parent);
  DexFuture  *(*find_by_uri)   (FoundryDocumentationProvider *self,
                                const char                   *uri);
  DexFuture  *(*list_bundles)  (FoundryDocumentationProvider *self);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo *foundry_documentation_provider_dup_plugin_info  (FoundryDocumentationProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel     *foundry_documentation_provider_list_roots       (FoundryDocumentationProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_documentation_provider_index            (FoundryDocumentationProvider *self,
                                                                 GListModel                   *roots);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_documentation_provider_query            (FoundryDocumentationProvider *self,
                                                                 FoundryDocumentationQuery    *query,
                                                                 FoundryDocumentationMatches  *matches);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_documentation_provider_list_children    (FoundryDocumentationProvider *self,
                                                                 FoundryDocumentation         *parent);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_documentation_provider_find_by_uri      (FoundryDocumentationProvider *self,
                                                                 const char                   *uri);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_documentation_provider_list_bundles     (FoundryDocumentationProvider *self);

G_END_DECLS
