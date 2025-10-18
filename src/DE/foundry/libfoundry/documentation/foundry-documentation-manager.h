/* foundry-documentation-manager.h
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

#include "foundry-service.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DOCUMENTATION_MANAGER (foundry_documentation_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryDocumentationManager, foundry_documentation_manager, FOUNDRY, DOCUMENTATION_MANAGER, FoundryService)

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_manager_query         (FoundryDocumentationManager *self,
                                                        FoundryDocumentationQuery   *query);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_manager_index         (FoundryDocumentationManager *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_documentation_manager_is_indexing   (FoundryDocumentationManager *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_manager_find_by_uri   (FoundryDocumentationManager *self,
                                                        const char                  *uri);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_manager_list_children (FoundryDocumentationManager *self,
                                                        FoundryDocumentation         *parent);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_manager_list_bundles  (FoundryDocumentationManager *self);

G_END_DECLS
