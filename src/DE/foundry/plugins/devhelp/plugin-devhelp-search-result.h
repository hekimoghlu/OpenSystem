/* plugin-devhelp-search-result.h
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

#include "plugin-devhelp-search-model.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_SEARCH_RESULT (plugin_devhelp_search_result_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpSearchResult, plugin_devhelp_search_result, PLUGIN, DEVHELP_SEARCH_RESULT, FoundryDocumentation)

struct _PluginDevhelpSearchResult
{
  FoundryDocumentation      parent_instance;
  GObject                  *item;
  guint                     position;
  PluginDevhelpSearchModel *model;
  GList                     link;
};

PluginDevhelpSearchResult *plugin_devhelp_search_result_new          (guint                      position);
guint                      plugin_devhelp_search_result_get_position (PluginDevhelpSearchResult *self);
gpointer                   plugin_devhelp_search_result_get_item     (PluginDevhelpSearchResult *self);
void                       plugin_devhelp_search_result_set_item     (PluginDevhelpSearchResult *self,
                                                                      gpointer                   item);

void                       plugin_devhelp_search_model_release       (PluginDevhelpSearchModel  *self,
                                                                      PluginDevhelpSearchResult *result);

G_END_DECLS
