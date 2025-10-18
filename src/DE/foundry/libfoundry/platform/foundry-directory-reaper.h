/* foundry-directory-reaper.h
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

#include <libdex.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIRECTORY_REAPER (foundry_directory_reaper_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDirectoryReaper, foundry_directory_reaper, FOUNDRY, DIRECTORY_REAPER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryDirectoryReaper *foundry_directory_reaper_new               (void);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_directory_reaper_add_directory     (FoundryDirectoryReaper  *self,
                                                                    GFile                   *directory,
                                                                    GTimeSpan                min_age);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_directory_reaper_add_file          (FoundryDirectoryReaper  *self,
                                                                    GFile                   *file,
                                                                    GTimeSpan                min_age);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_directory_reaper_add_glob          (FoundryDirectoryReaper  *self,
                                                                    GFile                   *directory,
                                                                    const gchar             *glob,
                                                                    GTimeSpan                min_age);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                foundry_directory_reaper_execute_sync      (FoundryDirectoryReaper  *self,
                                                                    GCancellable            *cancellable,
                                                                    GError                 **error);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_directory_reaper_execute_async     (FoundryDirectoryReaper   *self,
                                                                    GCancellable            *cancellable,
                                                                    GAsyncReadyCallback      callback,
                                                                    gpointer                 user_data);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                foundry_directory_reaper_execute_finish    (FoundryDirectoryReaper   *self,
                                                                    GAsyncResult            *result,
                                                                    GError                 **error);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_directory_reaper_execute           (FoundryDirectoryReaper   *self) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
