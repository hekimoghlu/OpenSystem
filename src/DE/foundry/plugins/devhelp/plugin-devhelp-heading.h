/* plugin-devhelp-heading.h
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
#include "plugin-devhelp-sdk.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_HEADING (plugin_devhelp_heading_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpHeading, plugin_devhelp_heading, PLUGIN, DEVHELP_HEADING, GomResource)

DexFuture  *plugin_devhelp_heading_find_by_uri     (PluginDevhelpRepository *repository,
                                                    const char              *uri);
gint64      plugin_devhelp_heading_get_id          (PluginDevhelpHeading    *self);
void        plugin_devhelp_heading_set_id          (PluginDevhelpHeading    *self,
                                                    gint64                   id);
gint64      plugin_devhelp_heading_get_book_id     (PluginDevhelpHeading    *self);
void        plugin_devhelp_heading_set_book_id     (PluginDevhelpHeading    *self,
                                                    gint64                   book_id);
gint64      plugin_devhelp_heading_get_parent_id   (PluginDevhelpHeading    *self);
void        plugin_devhelp_heading_set_parent_id   (PluginDevhelpHeading    *self,
                                                    gint64                   parent_id);
const char *plugin_devhelp_heading_get_title       (PluginDevhelpHeading    *self);
void        plugin_devhelp_heading_set_title       (PluginDevhelpHeading    *self,
                                                    const char              *title);
const char *plugin_devhelp_heading_get_uri         (PluginDevhelpHeading    *self);
void        plugin_devhelp_heading_set_uri         (PluginDevhelpHeading    *self,
                                                    const char              *uri);
gboolean    plugin_devhelp_heading_has_children    (PluginDevhelpHeading    *self);
DexFuture  *plugin_devhelp_heading_find_parent     (PluginDevhelpHeading    *self);
DexFuture  *plugin_devhelp_heading_find_sdk        (PluginDevhelpHeading    *self);
DexFuture  *plugin_devhelp_heading_find_book       (PluginDevhelpHeading    *self);
DexFuture  *plugin_devhelp_heading_list_headings   (PluginDevhelpHeading    *self);
DexFuture  *plugin_devhelp_heading_list_alternates (PluginDevhelpHeading    *self);

G_END_DECLS
