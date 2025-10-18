/* foundry-directory-item.h
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

#include <gio/gio.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIRECTORY_ITEM (foundry_directory_item_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDirectoryItem, foundry_directory_item, FOUNDRY, DIRECTORY_ITEM, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GFile                *foundry_directory_item_dup_directory     (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
GFile                *foundry_directory_item_dup_file          (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
GFileInfo            *foundry_directory_item_dup_info          (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_directory_item_dup_name          (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_directory_item_dup_display_name  (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
guint64               foundry_directory_item_get_size          (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
GFileType             foundry_directory_item_get_file_type     (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_directory_item_dup_content_type  (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon                *foundry_directory_item_dup_symbolic_icon (FoundryDirectoryItem *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean              foundry_directory_item_is_ignored        (FoundryDirectoryItem *self);
#ifdef FOUNDRY_FEATURE_VCS
FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsFileStatus  foundry_directory_item_get_status        (FoundryDirectoryItem *self);
#endif

G_END_DECLS
