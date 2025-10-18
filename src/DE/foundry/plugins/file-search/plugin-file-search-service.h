/* plugin-file-search-service.h
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

#define PLUGIN_TYPE_FILE_SEARCH_SERVICE (plugin_file_search_service_get_type())

G_DECLARE_FINAL_TYPE (PluginFileSearchService, plugin_file_search_service, PLUGIN, FILE_SEARCH_SERVICE, FoundryService)

DexFuture *plugin_file_search_service_query (PluginFileSearchService *self,
                                             const char              *search_text);

G_END_DECLS
