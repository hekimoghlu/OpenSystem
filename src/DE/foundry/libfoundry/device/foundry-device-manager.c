/* foundry-device-manager.c
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

#include <libpeas.h>

#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-device-manager.h"
#include "foundry-device-provider-private.h"
#include "foundry-device.h"
#include "foundry-model-manager.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

struct _FoundryDeviceManager
{
  FoundryService    parent_instance;
  GListModel       *flatten;
  PeasExtensionSet *addins;
  FoundryDevice    *device;
};

struct _FoundryDeviceManagerClass
{
  FoundryServiceClass parent_class;
};

enum {
  PROP_0,
  PROP_DEVICE,
  N_PROPS
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryDeviceManager, foundry_device_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_device_manager_provider_added (PeasExtensionSet *set,
                                       PeasPluginInfo   *plugin_info,
                                       GObject          *addin,
                                       gpointer          user_data)
{
  FoundryDeviceManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DEVICE_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DEVICE_MANAGER (self));

  g_debug ("Adding FoundryDeviceProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_device_provider_load (FOUNDRY_DEVICE_PROVIDER (addin)));
}

static void
foundry_device_manager_provider_removed (PeasExtensionSet *set,
                                         PeasPluginInfo   *plugin_info,
                                         GObject          *addin,
                                         gpointer          user_data)
{
  FoundryDeviceManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DEVICE_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DEVICE_MANAGER (self));

  g_debug ("Removing FoundryDeviceProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_device_provider_unload (FOUNDRY_DEVICE_PROVIDER (addin)));
}

static DexFuture *
foundry_device_manager_start_fiber (gpointer user_data)
{
  FoundryDeviceManager *self = user_data;
  g_autoptr(FoundrySettings) settings  = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryDevice) device = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autofree char *device_id = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_DEVICE_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  settings = foundry_context_load_project_settings (context);

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_device_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_device_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDeviceProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_device_provider_load (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  device_id = foundry_settings_get_string (settings, "device");

  if ((device = foundry_device_manager_find_device (self, device_id)))
    foundry_device_manager_set_device (self, device);

  return dex_future_new_true ();
}

static DexFuture *
foundry_device_manager_start (FoundryService *service)
{
  FoundryDeviceManager *self = (FoundryDeviceManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_DEVICE_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_device_manager_start_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_device_manager_stop (FoundryService *service)
{
  FoundryDeviceManager *self = (FoundryDeviceManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_device_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_device_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDeviceProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_device_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_device_manager_constructed (GObject *object)
{
  FoundryDeviceManager *self = (FoundryDeviceManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_device_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_DEVICE_PROVIDER,
                                         "context", context,
                                         NULL);

 g_object_set (self->flatten,
               "model", self->addins,
               NULL);
}

static void
foundry_device_manager_finalize (GObject *object)
{
  FoundryDeviceManager *self = (FoundryDeviceManager *)object;

  g_clear_object (&self->flatten);
  g_clear_object (&self->addins);
  g_clear_object (&self->device);

  G_OBJECT_CLASS (foundry_device_manager_parent_class)->finalize (object);
}

static void
foundry_device_manager_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryDeviceManager *self = FOUNDRY_DEVICE_MANAGER (object);

  switch (prop_id)
    {
    case PROP_DEVICE:
      g_value_take_object (value, foundry_device_manager_dup_device (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_device_manager_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryDeviceManager *self = FOUNDRY_DEVICE_MANAGER (object);

  switch (prop_id)
    {
    case PROP_DEVICE:
      foundry_device_manager_set_device (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_device_manager_class_init (FoundryDeviceManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_device_manager_constructed;
  object_class->finalize = foundry_device_manager_finalize;
  object_class->get_property = foundry_device_manager_get_property;
  object_class->set_property = foundry_device_manager_set_property;

  service_class->start = foundry_device_manager_start;
  service_class->stop = foundry_device_manager_stop;

  properties[PROP_DEVICE] =
    g_param_spec_object ("device", NULL, NULL,
                         FOUNDRY_TYPE_DEVICE,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_device_manager_init (FoundryDeviceManager *self)
{
  self->flatten = foundry_flatten_list_model_new (NULL);

  g_signal_connect_object (self->flatten,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

static GType
foundry_device_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DEVICE;
}

static guint
foundry_device_manager_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (G_LIST_MODEL (FOUNDRY_DEVICE_MANAGER (model)->flatten));
}

static gpointer
foundry_device_manager_get_item (GListModel *model,
                                 guint       position)
{
  return g_list_model_get_item (G_LIST_MODEL (FOUNDRY_DEVICE_MANAGER (model)->flatten), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_device_manager_get_item_type;
  iface->get_n_items = foundry_device_manager_get_n_items;
  iface->get_item = foundry_device_manager_get_item;
}

/**
 * foundry_device_manager_dup_device:
 * @self: a #FoundryDeviceManager
 *
 * Gets the active device to build for.
 *
 * Typically this is a #FoundryLocalDevice unless targeting a non-local device.
 *
 * Returns: (transfer full) (nullable): a #FoundryDevice or %NULL
 */
FoundryDevice *
foundry_device_manager_dup_device (FoundryDeviceManager *self)
{
  FoundryDevice *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DEVICE_MANAGER (self), NULL);

  g_set_object (&ret, self->device);

  return ret;
}

void
foundry_device_manager_set_device (FoundryDeviceManager *self,
                                   FoundryDevice        *device)
{
  g_autoptr(FoundryDevice) old = NULL;

  g_return_if_fail (FOUNDRY_IS_DEVICE_MANAGER (self));
  g_return_if_fail (!device || FOUNDRY_IS_DEVICE (device));

  if (self->device == device)
    return;

  if (device)
    g_object_ref (device);

  old = g_steal_pointer (&self->device);
  self->device = device;

  if (old != NULL)
    g_object_notify (G_OBJECT (old), "active");

  if (device != NULL)
    g_object_notify (G_OBJECT (device), "active");

  _foundry_contextual_invalidate_pipeline (FOUNDRY_CONTEXTUAL (self));
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_DEVICE]);
}

/**
 * foundry_device_manager_find_device:
 * @self: a #FoundryDeviceManager
 * @device_id: an identifier matching a #FoundryDevice:id
 *
 * Looks through available devices to find one matching @device_id.
 *
 * Returns: (transfer full) (nullable): a #FoundryDevice or %NULL
 */
FoundryDevice *
foundry_device_manager_find_device (FoundryDeviceManager *self,
                                    const char           *device_id)
{
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_DEVICE_MANAGER (self), NULL);
  g_return_val_if_fail (device_id != NULL, NULL);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDevice) device = g_list_model_get_item (G_LIST_MODEL (self), i);
      g_autofree char *id = foundry_device_dup_id (device);

      if (g_strcmp0 (device_id, id) == 0)
        return g_steal_pointer (&device);
    }

  return NULL;
}
