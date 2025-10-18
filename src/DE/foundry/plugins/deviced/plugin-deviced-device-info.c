/* plugin-deviced-device-info.c
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

#include "config.h"

#include "plugin-deviced-device-info.h"

struct _PluginDevicedDeviceInfo
{
  FoundryDeviceInfo parent_instance;
  DevdDevice *device;
  DevdClient *client;
};

G_DEFINE_FINAL_TYPE (PluginDevicedDeviceInfo, plugin_deviced_device_info, FOUNDRY_TYPE_DEVICE_INFO)

G_GNUC_BEGIN_IGNORE_DEPRECATIONS

static char *
plugin_deviced_device_info_dup_id (FoundryDeviceInfo *device_info)
{
  PluginDevicedDeviceInfo *self = (PluginDevicedDeviceInfo *)device_info;
  char *id = NULL;

  g_assert (PLUGIN_IS_DEVICED_DEVICE_INFO (self));
  g_assert (DEVD_IS_DEVICE (self->device));

  g_object_get (self->device, "id", &id, NULL);

  return id;
}

static char *
plugin_deviced_device_info_dup_name (FoundryDeviceInfo *device_info)
{
  PluginDevicedDeviceInfo *self = (PluginDevicedDeviceInfo *)device_info;

  g_assert (PLUGIN_IS_DEVICED_DEVICE_INFO (self));
  g_assert (DEVD_IS_CLIENT (self->client));

  return devd_client_get_name (self->client);
}

static FoundryTriplet *
plugin_deviced_device_info_dup_triplet (FoundryDeviceInfo *device_info)
{
  PluginDevicedDeviceInfo *self = (PluginDevicedDeviceInfo *)device_info;
  g_autoptr(DevdTriplet) triplet = NULL;

  g_assert (PLUGIN_IS_DEVICED_DEVICE_INFO (self));
  g_assert (DEVD_IS_CLIENT (self->client));

  triplet = devd_client_get_triplet (self->client);

  return foundry_triplet_new_with_quadruplet (devd_triplet_get_arch (triplet),
                                              devd_triplet_get_vendor (triplet),
                                              devd_triplet_get_kernel (triplet),
                                              devd_triplet_get_operating_system (triplet));
}

static void
plugin_deviced_device_info_finalize (GObject *object)
{
  PluginDevicedDeviceInfo *self = (PluginDevicedDeviceInfo *)object;

  g_clear_object (&self->client);
  g_clear_object (&self->device);

  G_OBJECT_CLASS (plugin_deviced_device_info_parent_class)->finalize (object);
}

static void
plugin_deviced_device_info_class_init (PluginDevicedDeviceInfoClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDeviceInfoClass *device_info_class = FOUNDRY_DEVICE_INFO_CLASS (klass);

  object_class->finalize = plugin_deviced_device_info_finalize;

  device_info_class->dup_id = plugin_deviced_device_info_dup_id;
  device_info_class->dup_name = plugin_deviced_device_info_dup_name;
  device_info_class->dup_triplet = plugin_deviced_device_info_dup_triplet;
}

static void
plugin_deviced_device_info_init (PluginDevicedDeviceInfo *self)
{
}

static DexFuture *
plugin_deviced_device_info_new_fiber (gpointer data)
{
  FoundryPair *pair = data;
  DevdDevice *device = DEVD_DEVICE (pair->first);
  DevdClient *client = DEVD_CLIENT (pair->second);
  PluginDevicedDeviceInfo *self;

  g_assert (DEVD_IS_DEVICE (device));
  g_assert (DEVD_IS_CLIENT (client));

  self = g_object_new (PLUGIN_TYPE_DEVICED_DEVICE_INFO, NULL);
  self->device = g_object_ref (device);
  self->client = g_object_ref (client);

  return dex_future_new_take_object (g_steal_pointer (&self));
}

DexFuture *
plugin_deviced_device_info_new (DevdDevice *device,
                                DevdClient *client)
{
  dex_return_error_if_fail (DEVD_IS_DEVICE (device));
  dex_return_error_if_fail (DEVD_IS_CLIENT (client));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_deviced_device_info_new_fiber,
                              foundry_pair_new (device, client),
                              (GDestroyNotify) foundry_pair_free);
}

G_GNUC_END_IGNORE_DEPRECATIONS
