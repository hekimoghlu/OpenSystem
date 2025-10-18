/* foundry-file.h
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

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_file_find_in_ancestors       (GFile        *file,
                                                 const char   *name) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_file_find_with_depth         (GFile        *file,
                                                 const gchar  *pattern,
                                                 guint         max_depth) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_file_find_regex_with_depth   (GFile        *file,
                                                 GRegex       *regex,
                                                 guint         max_depth) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_file_query_exists_nofollow   (GFile        *file) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
GFile     *foundry_file_canonicalize            (GFile        *file,
                                                 GError      **error) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_file_is_in                   (GFile        *file,
                                                 GFile        *toplevel);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_file_list_children_typed     (GFile        *file,
                                                 GFileType     file_type,
                                                 const char   *attributes) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_host_file_get_contents_bytes (const char   *path) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
