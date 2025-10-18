/* plugin-deviced-device-provider.c
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

#include <libdeviced.h>

#include "plugin-deviced-device.h"
#include "plugin-deviced-device-provider.h"

struct _PluginDevicedDeviceProvider
{
  FoundryDeviceProvider  parent_instance;
  DevdBrowser           *browser;
};

G_DEFINE_FINAL_TYPE (PluginDevicedDeviceProvider, plugin_deviced_device_provider, FOUNDRY_TYPE_DEVICE_PROVIDER)

static void
plugin_deviced_device_provider_device_added_cb (PluginDevicedDeviceProvider *self,
                                                DevdDevice                  *device,
                                                DevdBrowser                 *browser)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryDevice) wrapped = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_DEVICED_DEVICE_PROVIDER (self));
  g_assert (DEVD_IS_DEVICE (device));
  g_assert (DEVD_IS_BROWSER (browser));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  wrapped = g_object_new (PLUGIN_TYPE_DEVICED_DEVICE,
                          "context", context,
                          "device", device,
                          NULL);

  foundry_device_provider_device_added (FOUNDRY_DEVICE_PROVIDER (self), wrapped);
}

static void
plugin_deviced_device_provider_device_removed_cb (PluginDevicedDeviceProvider *self,
                                                  DevdDevice                  *device,
                                                  DevdBrowser                 *browser)
{
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_DEVICED_DEVICE_PROVIDER (self));
  g_assert (DEVD_IS_DEVICE (device));
  g_assert (DEVD_IS_BROWSER (browser));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PluginDevicedDevice) wrapped = g_list_model_get_item (G_LIST_MODEL (self), i);
      g_autoptr(DevdDevice) wrapped_device = plugin_deviced_device_dup_device (wrapped);

      if (wrapped_device == device)
        {
          foundry_device_provider_device_removed (FOUNDRY_DEVICE_PROVIDER (self),
                                                  FOUNDRY_DEVICE (wrapped));
          break;
        }
    }
}

static void
plugin_deviced_device_provider_load_cb (GObject      *object,
                                        GAsyncResult *result,
                                        gpointer      user_data)
{
  DevdBrowser *browser = (DevdBrowser *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (DEVD_IS_BROWSER (browser));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!devd_browser_load_finish (browser, result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static DexFuture *
plugin_deviced_device_provider_load (FoundryDeviceProvider *device_provider)
{
  PluginDevicedDeviceProvider *self = (PluginDevicedDeviceProvider *)device_provider;
  DexPromise *promise;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_DEVICED_DEVICE_PROVIDER (self));

  promise = dex_promise_new_cancellable ();

  self->browser = devd_browser_new ();

  g_signal_connect_object (self->browser,
                           "device-added",
                           G_CALLBACK (plugin_deviced_device_provider_device_added_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->browser,
                           "device-removed",
                           G_CALLBACK (plugin_deviced_device_provider_device_removed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  devd_browser_load_async (self->browser,
                           dex_promise_get_cancellable (promise),
                           plugin_deviced_device_provider_load_cb,
                           dex_ref (promise));

  return DEX_FUTURE (promise);
}

static DexFuture *
plugin_deviced_device_provider_unload (FoundryDeviceProvider *device_provider)
{
  PluginDevicedDeviceProvider *self = (PluginDevicedDeviceProvider *)device_provider;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_DEVICED_DEVICE_PROVIDER (self));

  g_signal_handlers_disconnect_by_func (self->browser,
                                        G_CALLBACK (plugin_deviced_device_provider_device_added_cb),
                                        self);
  g_signal_handlers_disconnect_by_func (self->browser,
                                        G_CALLBACK (plugin_deviced_device_provider_device_removed_cb),
                                        self);
  g_clear_object (&self->browser);

  return dex_future_new_true ();
}

static char *
plugin_deviced_device_provider_dup_name (FoundryDeviceProvider *device_provider)
{
  return g_strdup ("deviced");
}

static void
plugin_deviced_device_provider_finalize (GObject *object)
{
  PluginDevicedDeviceProvider *self = (PluginDevicedDeviceProvider *)object;

  g_clear_object (&self->browser);

  G_OBJECT_CLASS (plugin_deviced_device_provider_parent_class)->finalize (object);
}

static void
plugin_deviced_device_provider_class_init (PluginDevicedDeviceProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDeviceProviderClass *device_provider_class = FOUNDRY_DEVICE_PROVIDER_CLASS (klass);

  object_class->finalize = plugin_deviced_device_provider_finalize;

  device_provider_class->load = plugin_deviced_device_provider_load;
  device_provider_class->unload = plugin_deviced_device_provider_unload;
  device_provider_class->dup_name = plugin_deviced_device_provider_dup_name;
}

static void
plugin_deviced_device_provider_init (PluginDevicedDeviceProvider *self)
{
}
