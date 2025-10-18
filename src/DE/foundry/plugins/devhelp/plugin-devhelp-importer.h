/* plugin-devhelp-importer.h
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

#include <libdex.h>

#include "plugin-devhelp-progress.h"
#include "plugin-devhelp-repository.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_IMPORTER (plugin_devhelp_importer_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpImporter, plugin_devhelp_importer, PLUGIN, DEVHELP_IMPORTER, GObject)

PluginDevhelpImporter *plugin_devhelp_importer_new           (void);
guint                  plugin_devhelp_importer_get_size      (PluginDevhelpImporter   *self);
void                   plugin_devhelp_importer_add_directory (PluginDevhelpImporter   *self,
                                                              const char              *directory,
                                                              gint64                   sdk_id);
void                   plugin_devhelp_importer_set_sdk_id    (PluginDevhelpImporter   *self,
                                                              gint64                   sdk_id);
DexFuture             *plugin_devhelp_importer_import        (PluginDevhelpImporter   *self,
                                                              PluginDevhelpRepository *repository,
                                                              PluginDevhelpProgress   *progress);

G_END_DECLS
