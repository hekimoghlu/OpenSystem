/* plugin-deviced-device.c
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

#include "plugin-deviced-device.h"
#include "plugin-deviced-device-info.h"
#include "plugin-deviced-dex.h"

struct _PluginDevicedDevice
{
  FoundryDevice  parent_instance;
  DevdDevice    *device;
  DexFuture     *client;
};

enum {
  PROP_0,
  PROP_DEVICE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginDevicedDevice, plugin_deviced_device, FOUNDRY_TYPE_DEVICE)

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_deviced_device_load_info_cb (DexFuture *completed,
                                    gpointer   user_data)
{
  g_autoptr(DevdClient) client = dex_await_object (dex_ref (completed), NULL);
  PluginDevicedDevice *self = user_data;

  g_assert (DEVD_IS_CLIENT (client));
  g_assert (PLUGIN_IS_DEVICED_DEVICE (self));

  return plugin_deviced_device_info_new (self->device, client);
}

static DexFuture *
plugin_deviced_device_load_info (FoundryDevice *device)
{
  PluginDevicedDevice *self = (PluginDevicedDevice *)device;
  DexFuture *future;

  g_assert (PLUGIN_IS_DEVICED_DEVICE (self));

  future = plugin_deviced_device_load_client (self);
  future = dex_future_then (future,
                            plugin_deviced_device_load_info_cb,
                            g_object_ref (self),
                            g_object_unref);

  return future;
}

static char *
plugin_deviced_device_dup_id (FoundryDevice *device)
{
  PluginDevicedDevice *self = (PluginDevicedDevice *)device;
  g_autofree char *id = NULL;

  g_assert (PLUGIN_IS_DEVICED_DEVICE (self));

  g_object_get (self->device, "id", &id, NULL);

  return g_steal_pointer (&id);
}

static void
plugin_deviced_device_finalize (GObject *object)
{
  PluginDevicedDevice *self = (PluginDevicedDevice *)object;

  g_clear_object (&self->device);
  dex_clear (&self->client);

  G_OBJECT_CLASS (plugin_deviced_device_parent_class)->finalize (object);
}

static void
plugin_deviced_device_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  PluginDevicedDevice *self = PLUGIN_DEVICED_DEVICE (object);

  switch (prop_id)
    {
    case PROP_DEVICE:
      g_value_take_object (value, plugin_deviced_device_dup_device (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_deviced_device_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  PluginDevicedDevice *self = PLUGIN_DEVICED_DEVICE (object);

  switch (prop_id)
    {
    case PROP_DEVICE:
      self->device = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_deviced_device_class_init (PluginDevicedDeviceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDeviceClass *device_class = FOUNDRY_DEVICE_CLASS (klass);

  object_class->finalize = plugin_deviced_device_finalize;
  object_class->get_property = plugin_deviced_device_get_property;
  object_class->set_property = plugin_deviced_device_set_property;

  device_class->dup_id = plugin_deviced_device_dup_id;
  device_class->load_info = plugin_deviced_device_load_info;

  properties[PROP_DEVICE] =
    g_param_spec_object ("device", NULL, NULL,
                         DEVD_TYPE_DEVICE,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_deviced_device_init (PluginDevicedDevice *self)
{
}

DevdDevice *
plugin_deviced_device_dup_device (PluginDevicedDevice *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVICED_DEVICE (self), NULL);

  return g_object_ref (self->device);
}

DexFuture *
plugin_deviced_device_load_client (PluginDevicedDevice *self)
{
  dex_return_error_if_fail (PLUGIN_IS_DEVICED_DEVICE (self));
  dex_return_error_if_fail (DEVD_IS_DEVICE (self->device));
  dex_return_error_if_fail (!self->client || DEX_IS_FUTURE (self->client));

  if (self->client == NULL)
    {
      g_autoptr(DevdClient) client = devd_device_create_client (self->device);
      self->client = devd_client_connect (client);
    }

  return dex_ref (DEX_FUTURE (self->client));
}

static void
_devd_client_list_apps_cb (GObject      *object,
                           GAsyncResult *result,
                           gpointer      user_data)
{
  DevdClient *client = (DevdClient *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GPtrArray) apps = NULL;
  g_autoptr(GError) error = NULL;

  if (!(apps = devd_client_list_apps_finish (client, result, &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    {
      g_auto(GValue) value = G_VALUE_INIT;
      g_value_init (&value, G_TYPE_PTR_ARRAY);
      g_value_take_boxed (&value, g_steal_pointer (&apps));
      dex_promise_resolve (promise, &value);
    }
}

static DexFuture *
_devd_client_list_apps (DevdClient *client)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  devd_client_list_apps_async (client,
                               dex_promise_get_cancellable (promise),
                               _devd_client_list_apps_cb,
                               dex_ref (promise));
  return DEX_FUTURE (promise);
}

static DexFuture *
plugin_deviced_device_query_commit_fiber (PluginDevicedDevice *self,
                                          const char          *app_id)
{
  g_autoptr(DevdClient) client = NULL;
  g_autoptr(GPtrArray) apps = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_DEVICED_DEVICE (self));
  g_assert (app_id != NULL);

  if (!(client = dex_await_object (plugin_deviced_device_load_client (self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(apps = dex_await_boxed (_devd_client_list_apps (client), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  for (guint i = 0; i < apps->len; i++)
    {
      DevdAppInfo *app_info = g_ptr_array_index (apps, i);

      if (g_strcmp0 (app_id, devd_app_info_get_id (app_info)) == 0)
        {
          const char *commit_id = devd_app_info_get_commit_id (app_info);

          if (commit_id != NULL)
            return dex_future_new_take_string (g_strdup (commit_id));
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Failed to locate commit");
}

DexFuture *
plugin_deviced_device_query_commit (PluginDevicedDevice *self,
                                    const char          *app_id)
{
  dex_return_error_if_fail (PLUGIN_IS_DEVICED_DEVICE (self));
  dex_return_error_if_fail (app_id);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_deviced_device_query_commit_fiber),
                                  2,
                                  PLUGIN_TYPE_DEVICED_DEVICE, self,
                                  G_TYPE_STRING, app_id);
}

typedef struct _InstallBundle
{
  PluginDevicedDevice   *self;
  char                  *bundle_path;
  GFileProgressCallback  progress;
  gpointer               progress_data;
  GDestroyNotify         progress_data_destroy;
} InstallBundle;

static void
install_bundle_free (InstallBundle *state)
{
  g_clear_object (&state->self);
  g_clear_pointer (&state->bundle_path, g_free);

  if (state->progress_data_destroy)
    {
      state->progress_data_destroy (state->progress_data);
      state->progress_data = NULL;
    }

  g_free (state);
}

static DexFuture *
plugin_deviced_device_install_bundle_fiber (gpointer data)
{
  InstallBundle *state = data;
  g_autoptr(DevdTransferService) transfer = NULL;
  g_autoptr(DevdFlatpakService) flatpak = NULL;
  g_autoptr(DevdClient) client = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) our_file = NULL;
  g_autofree char *name = NULL;
  g_autofree char *their_file = NULL;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_DEVICED_DEVICE (state->self));
  g_assert (state->bundle_path != NULL);

  if (!(client = dex_await_object (plugin_deviced_device_load_client (state->self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(transfer = devd_transfer_service_new (client, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  name = g_path_get_basename (state->bundle_path);
  their_file = g_build_filename (".cache", "deviced", name, NULL);
  our_file = g_file_new_for_path (state->bundle_path);

  if (!dex_await (devd_transfer_service_put_file (transfer,
                                                  our_file,
                                                  their_file,
                                                  g_steal_pointer (&state->progress),
                                                  g_steal_pointer (&state->progress_data),
                                                  g_steal_pointer (&state->progress_data_destroy)),
                  &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(flatpak = devd_flatpak_service_new (client, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!dex_await (devd_flatpak_service_install_bundle (flatpak, their_file), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

DexFuture *
plugin_deviced_device_install_bundle (PluginDevicedDevice   *self,
                                      const char            *bundle_path,
                                      GFileProgressCallback  progress,
                                      gpointer               progress_data,
                                      GDestroyNotify         progress_data_destroy)
{
  InstallBundle *state;

  dex_return_error_if_fail (PLUGIN_IS_DEVICED_DEVICE (self));
  dex_return_error_if_fail (bundle_path != NULL);

  state = g_new0 (InstallBundle, 1);
  state->self = g_object_ref (self);
  state->bundle_path = g_strdup (bundle_path);
  state->progress = progress;
  state->progress_data = progress_data;
  state->progress_data_destroy = progress_data_destroy;

  return dex_scheduler_spawn (NULL, 0,
                              plugin_deviced_device_install_bundle_fiber,
                              state,
                              (GDestroyNotify) install_bundle_free);
}

char *
plugin_deviced_device_dup_network_address (PluginDevicedDevice  *self,
                                           guint                *port,
                                           GError              **error)
{
  GInetSocketAddress *address;

  g_return_val_if_fail (PLUGIN_IS_DEVICED_DEVICE (self), NULL);
  g_return_val_if_fail (DEVD_IS_DEVICE (self->device), NULL);
  g_return_val_if_fail (port != NULL, NULL);

  if (!DEVD_IS_NETWORK_DEVICE (self->device) ||
      !(address = devd_network_device_get_address (DEVD_NETWORK_DEVICE (self->device))))
  {
    g_set_error_literal (error,
                         G_IO_ERROR,
                         G_IO_ERROR_FAILED,
                         "Not configured for deviced communication");
    return NULL;
  }

  *port = g_inet_socket_address_get_port (address);

  return g_inet_address_to_string (g_inet_socket_address_get_address (address));
}
