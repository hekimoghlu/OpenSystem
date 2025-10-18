/* foundry-documentation-matches.h
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

#include <libdex.h>

#include "foundry-version-macros.h"
#include "foundry-types.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DOCUMENTATION_MATCHES (foundry_documentation_matches_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDocumentationMatches, foundry_documentation_matches, FOUNDRY, DOCUMENTATION_MATCHES, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryDocumentationQuery *foundry_documentation_matches_dup_query     (FoundryDocumentationMatches *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel                *foundry_documentation_matches_list_sections (FoundryDocumentationMatches *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_documentation_matches_add_section   (FoundryDocumentationMatches *self,
                                                                        GListModel                  *section);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_documentation_matches_await         (FoundryDocumentationMatches *self);

G_END_DECLS
