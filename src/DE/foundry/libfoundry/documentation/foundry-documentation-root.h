/* foundry-documentation-root.h
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

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DOCUMENTATION_ROOT (foundry_documentation_root_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDocumentationRoot, foundry_documentation_root, FOUNDRY, DOCUMENTATION_ROOT, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryDocumentationRoot *foundry_documentation_root_new              (const char               *identifier,
                                                                       const char               *title,
                                                                       const char               *version,
                                                                       GIcon                    *icon,
                                                                       GListModel               *directories);
FOUNDRY_AVAILABLE_IN_ALL
char                     *foundry_documentation_root_dup_title        (FoundryDocumentationRoot *self);
FOUNDRY_AVAILABLE_IN_ALL
char                     *foundry_documentation_root_dup_identifier   (FoundryDocumentationRoot *self);
FOUNDRY_AVAILABLE_IN_ALL
char                     *foundry_documentation_root_dup_version      (FoundryDocumentationRoot *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon                    *foundry_documentation_root_dup_icon         (FoundryDocumentationRoot *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel               *foundry_documentation_root_list_directories (FoundryDocumentationRoot *self);

G_END_DECLS
