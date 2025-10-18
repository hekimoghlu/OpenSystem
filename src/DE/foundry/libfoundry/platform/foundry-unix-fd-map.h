/* foundry-unix-fd-map.h
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

#if !defined (FOUNDRY_INSIDE) && !defined (FOUNDRY_COMPILATION)
# error "Only <foundry.h> can be included directly."
#endif

#include <gio/gio.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_UNIX_FD_MAP (foundry_unix_fd_map_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryUnixFDMap, foundry_unix_fd_map, FOUNDRY, UNIX_FD_MAP, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryUnixFDMap *foundry_unix_fd_map_new             (void);
FOUNDRY_AVAILABLE_IN_ALL
guint             foundry_unix_fd_map_get_length      (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_peek_stdin      (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_peek_stdout     (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_peek_stderr     (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_steal_stdin     (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_steal_stdout    (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_steal_stderr    (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_unix_fd_map_steal_from      (FoundryUnixFDMap  *self,
                                                       FoundryUnixFDMap  *other,
                                                       GError           **error);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_peek            (FoundryUnixFDMap  *self,
                                                       guint              index,
                                                       int               *dest_fd);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_get             (FoundryUnixFDMap  *self,
                                                       guint              index,
                                                       int               *dest_fd,
                                                       GError           **error);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_steal           (FoundryUnixFDMap  *self,
                                                       guint              index,
                                                       int               *dest_fd);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_unix_fd_map_take            (FoundryUnixFDMap  *self,
                                                       int                source_fd,
                                                       int                dest_fd);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_unix_fd_map_open_file       (FoundryUnixFDMap  *self,
                                                       const char        *filename,
                                                       int                mode,
                                                       int                dest_fd,
                                                       GError           **error);
FOUNDRY_AVAILABLE_IN_ALL
int               foundry_unix_fd_map_get_max_dest_fd (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_unix_fd_map_stdin_isatty    (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_unix_fd_map_stdout_isatty   (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_unix_fd_map_stderr_isatty   (FoundryUnixFDMap  *self);
FOUNDRY_AVAILABLE_IN_ALL
GIOStream        *foundry_unix_fd_map_create_stream   (FoundryUnixFDMap  *self,
                                                       int                dest_read_fd,
                                                       int                dest_write_fd,
                                                       GError           **error);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_unix_fd_map_silence_fd      (FoundryUnixFDMap  *self,
                                                       int                dest_fd,
                                                       GError           **error);

G_END_DECLS
