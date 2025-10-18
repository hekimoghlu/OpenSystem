/* plugin-devhelp-keyword.h
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

#include "plugin-devhelp-repository.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_KEYWORD (plugin_devhelp_keyword_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpKeyword, plugin_devhelp_keyword, PLUGIN, DEVHELP_KEYWORD, GomResource)

DexFuture  *plugin_devhelp_keyword_find_by_uri     (PluginDevhelpRepository *repository,
                                                    const char              *uri);
DexFuture  *plugin_devhelp_keyword_find_book       (PluginDevhelpKeyword    *self);
gint64      plugin_devhelp_keyword_get_id          (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_id          (PluginDevhelpKeyword    *self,
                                                    gint64                   id);
gint64      plugin_devhelp_keyword_get_book_id     (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_book_id     (PluginDevhelpKeyword    *self,
                                                    gint64                   book_id);
const char *plugin_devhelp_keyword_get_kind        (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_kind        (PluginDevhelpKeyword    *self,
                                                    const char              *kind);
const char *plugin_devhelp_keyword_get_since       (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_since       (PluginDevhelpKeyword    *self,
                                                    const char              *since);
const char *plugin_devhelp_keyword_get_stability   (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_stability   (PluginDevhelpKeyword    *self,
                                                    const char              *stability);
const char *plugin_devhelp_keyword_get_deprecated  (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_deprecated  (PluginDevhelpKeyword    *self,
                                                    const char              *deprecated);
const char *plugin_devhelp_keyword_get_name        (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_name        (PluginDevhelpKeyword    *self,
                                                    const char              *name);
const char *plugin_devhelp_keyword_get_uri         (PluginDevhelpKeyword    *self);
void        plugin_devhelp_keyword_set_uri         (PluginDevhelpKeyword    *self,
                                                    const char              *uri);
DexFuture  *plugin_devhelp_keyword_list_alternates (PluginDevhelpKeyword    *self);
char       *plugin_devhelp_keyword_query_attribute (PluginDevhelpKeyword    *self,
                                                    const char              *attribute);

G_END_DECLS
