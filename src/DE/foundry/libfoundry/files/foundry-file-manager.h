/* foundry-file-manager.h
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

#pragma once

#include "foundry-service.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FILE_MANAGER (foundry_file_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryFileManager, foundry_file_manager, FOUNDRY, FILE_MANAGER, FoundryService)

FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_file_manager_show               (FoundryFileManager *self,
                                                     GFile              *file) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
GIcon      *foundry_file_manager_find_symbolic_icon (FoundryFileManager *self,
                                                     const char         *content_type,
                                                     const char         *filename) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_file_manager_write_metadata     (FoundryFileManager *self,
                                                     GFile              *file,
                                                     GFileInfo          *file_info) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_file_manager_read_metadata      (FoundryFileManager *self,
                                                     GFile              *file,
                                                     const char         *attributes) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_file_manager_guess_language     (FoundryFileManager *self,
                                                     GFile              *file,
                                                     const char         *content_type,
                                                     GBytes             *contents) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
char      **foundry_file_manager_list_languages     (FoundryFileManager *self) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
