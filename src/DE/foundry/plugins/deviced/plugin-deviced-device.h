/* plugin-deviced-device.h
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
#include <libdeviced.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVICED_DEVICE (plugin_deviced_device_get_type())

G_DECLARE_FINAL_TYPE (PluginDevicedDevice, plugin_deviced_device, PLUGIN, DEVICED_DEVICE, FoundryDevice)

DevdDevice *plugin_deviced_device_dup_device          (PluginDevicedDevice    *self);
DexFuture  *plugin_deviced_device_load_client         (PluginDevicedDevice    *self) G_GNUC_WARN_UNUSED_RESULT;
DexFuture  *plugin_deviced_device_query_commit        (PluginDevicedDevice    *self,
                                                       const char             *app_id) G_GNUC_WARN_UNUSED_RESULT;
DexFuture  *plugin_deviced_device_install_bundle      (PluginDevicedDevice    *self,
                                                       const char             *bundle_path,
                                                       GFileProgressCallback   progress,
                                                       gpointer                progress_data,
                                                       GDestroyNotify          progress_data_destroy) G_GNUC_WARN_UNUSED_RESULT;
char       *plugin_deviced_device_dup_network_address (PluginDevicedDevice    *self,
                                                       guint                  *port,
                                                       GError                **error);

G_END_DECLS
