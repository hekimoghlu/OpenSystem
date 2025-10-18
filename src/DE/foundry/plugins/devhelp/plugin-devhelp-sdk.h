/* plugin-devhelp-sdk.h
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

#define PLUGIN_TYPE_DEVHELP_SDK (plugin_devhelp_sdk_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpSdk, plugin_devhelp_sdk, PLUGIN, DEVHELP_SDK, GomResource)

gint64      plugin_devhelp_sdk_get_id         (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_id         (PluginDevhelpSdk *self,
                                               gint64            id);
const char *plugin_devhelp_sdk_get_kind       (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_kind       (PluginDevhelpSdk *self,
                                               const char       *kind);
const char *plugin_devhelp_sdk_get_name       (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_name       (PluginDevhelpSdk *self,
                                               const char       *name);
const char *plugin_devhelp_sdk_get_version    (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_version    (PluginDevhelpSdk *self,
                                               const char       *version);
const char *plugin_devhelp_sdk_get_online_uri (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_online_uri (PluginDevhelpSdk *self,
                                               const char       *online_uri);
char       *plugin_devhelp_sdk_dup_title      (PluginDevhelpSdk *self);
const char *plugin_devhelp_sdk_get_ident      (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_ident      (PluginDevhelpSdk *self,
                                               const char       *ident);
const char *plugin_devhelp_sdk_get_icon_name  (PluginDevhelpSdk *self);
void        plugin_devhelp_sdk_set_icon_name  (PluginDevhelpSdk *self,
                                               const char       *icon_name);
DexFuture  *plugin_devhelp_sdk_list_books     (PluginDevhelpSdk *self);

G_END_DECLS
