/* plugin-devhelp-navigatable.h
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

#include <foundry.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_NAVIGATABLE (plugin_devhelp_navigatable_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpNavigatable, plugin_devhelp_navigatable, PLUGIN, DEVHELP_NAVIGATABLE, FoundryDocumentation)

PluginDevhelpNavigatable *plugin_devhelp_navigatable_new              (void);
PluginDevhelpNavigatable *plugin_devhelp_navigatable_new_for_resource (GObject                  *resource);
GIcon                    *plugin_devhelp_navigatable_get_icon         (PluginDevhelpNavigatable *self);
void                      plugin_devhelp_navigatable_set_icon         (PluginDevhelpNavigatable *self,
                                                                       GIcon                    *icon);
const char               *plugin_devhelp_navigatable_get_title        (PluginDevhelpNavigatable *self);
void                      plugin_devhelp_navigatable_set_title        (PluginDevhelpNavigatable *self,
                                                                       const char               *title);
GIcon                    *plugin_devhelp_navigatable_get_menu_icon    (PluginDevhelpNavigatable *self);
void                      plugin_devhelp_navigatable_set_menu_icon    (PluginDevhelpNavigatable *self,
                                                                       GIcon                    *menu_icon);
const char               *plugin_devhelp_navigatable_get_menu_title   (PluginDevhelpNavigatable *self);
void                      plugin_devhelp_navigatable_set_menu_title   (PluginDevhelpNavigatable *self,
                                                                       const char               *menu_title);
const char               *plugin_devhelp_navigatable_get_uri          (PluginDevhelpNavigatable *self);
void                      plugin_devhelp_navigatable_set_uri          (PluginDevhelpNavigatable *self,
                                                                       const char               *uri);
gpointer                  plugin_devhelp_navigatable_get_item         (PluginDevhelpNavigatable *self);
void                      plugin_devhelp_navigatable_set_item         (PluginDevhelpNavigatable *self,
                                                                       gpointer                  item);

G_END_DECLS
