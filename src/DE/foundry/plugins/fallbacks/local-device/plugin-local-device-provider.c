/* plugin-local-device-provider.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "plugin-local-device-provider.h"

struct _PluginLocalDeviceProvider
{
  FoundryDeviceProvider  parent_instance;
  FoundryDevice         *device;
};

G_DEFINE_FINAL_TYPE (PluginLocalDeviceProvider, plugin_local_device_provider, FOUNDRY_TYPE_DEVICE_PROVIDER)

static DexFuture *
plugin_local_device_provider_load (FoundryDeviceProvider *provider)
{
  PluginLocalDeviceProvider *self = (PluginLocalDeviceProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_LOCAL_DEVICE_PROVIDER (self));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      self->device = foundry_local_device_new (context);
      foundry_device_provider_device_added (FOUNDRY_DEVICE_PROVIDER (self), self->device);
    }

  FOUNDRY_RETURN (dex_future_new_true ());
}

static DexFuture *
plugin_local_device_provider_unload (FoundryDeviceProvider *provider)
{
  PluginLocalDeviceProvider *self = (PluginLocalDeviceProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_LOCAL_DEVICE_PROVIDER (self));

  if (self->device != NULL)
    {
      foundry_device_provider_device_removed (FOUNDRY_DEVICE_PROVIDER (self), self->device);
      g_clear_object (&self->device);
    }

  FOUNDRY_RETURN (dex_future_new_true ());
}

static void
plugin_local_device_provider_finalize (GObject *object)
{
  PluginLocalDeviceProvider *self = (PluginLocalDeviceProvider *)object;

  g_clear_object (&self->device);

  G_OBJECT_CLASS (plugin_local_device_provider_parent_class)->finalize (object);
}

static void
plugin_local_device_provider_class_init (PluginLocalDeviceProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDeviceProviderClass *device_provider_class = FOUNDRY_DEVICE_PROVIDER_CLASS (klass);

  object_class->finalize = plugin_local_device_provider_finalize;

  device_provider_class->load = plugin_local_device_provider_load;
  device_provider_class->unload = plugin_local_device_provider_unload;
}

static void
plugin_local_device_provider_init (PluginLocalDeviceProvider *self)
{
}
