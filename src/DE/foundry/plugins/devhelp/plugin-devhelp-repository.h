/* plugin-devhelp-repository.h
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

#include <gom/gom.h>
#include <libdex.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_REPOSITORY (plugin_devhelp_repository_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpRepository, plugin_devhelp_repository, PLUGIN, DEVHELP_REPOSITORY, GomRepository)

DexFuture  *plugin_devhelp_repository_open                  (const char              *path);
DexFuture  *plugin_devhelp_repository_close                 (PluginDevhelpRepository *self);
DexFuture  *plugin_devhelp_repository_list                  (PluginDevhelpRepository *self,
                                                             GType                    resource_type,
                                                             GomFilter               *filter);
DexFuture  *plugin_devhelp_repository_list_sorted           (PluginDevhelpRepository *self,
                                                             GType                    resource_type,
                                                             GomFilter               *filter,
                                                             GomSorting              *sorting);
DexFuture  *plugin_devhelp_repository_count                 (PluginDevhelpRepository *self,
                                                             GType                    resource_type,
                                                             GomFilter               *filter);
DexFuture  *plugin_devhelp_repository_find_one              (PluginDevhelpRepository *self,
                                                             GType                    resource_type,
                                                             GomFilter               *filter);
DexFuture  *plugin_devhelp_repository_list_sdks             (PluginDevhelpRepository *self);
DexFuture  *plugin_devhelp_repository_list_sdks_by_newest   (PluginDevhelpRepository *self);
DexFuture  *plugin_devhelp_repository_delete                (PluginDevhelpRepository *self,
                                                             GType                    resource_type,
                                                             GomFilter               *filter);
DexFuture  *plugin_devhelp_repository_find_sdk              (PluginDevhelpRepository *self,
                                                             const char              *ident);
const char *plugin_devhelp_repository_get_cached_book_title (PluginDevhelpRepository *self,
                                                             gint64                   book_id);
const char *plugin_devhelp_repository_get_cached_sdk_title  (PluginDevhelpRepository *self,
                                                             gint64                   sdk_id);
gint64      plugin_devhelp_repository_get_cached_sdk_id     (PluginDevhelpRepository *self,
                                                             gint64                   book_id);

G_END_DECLS
