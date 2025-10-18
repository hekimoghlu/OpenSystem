/* plugin-deviced-device-info.h
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

#define PLUGIN_TYPE_DEVICED_DEVICE_INFO (plugin_deviced_device_info_get_type())

G_DECLARE_FINAL_TYPE (PluginDevicedDeviceInfo, plugin_deviced_device_info, PLUGIN, DEVICED_DEVICE_INFO, FoundryDeviceInfo)

DexFuture *plugin_deviced_device_info_new (DevdDevice *device,
                                           DevdClient *client);

G_END_DECLS
