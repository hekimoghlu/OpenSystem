/* foundry-device-provider.c
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

#include <glib/gi18n-lib.h>

#include "foundry-device-provider-private.h"
#include "foundry-device-private.h"

typedef struct
{
  GListStore *store;
} FoundryDeviceProviderPrivate;

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundryDeviceProvider, foundry_device_provider, FOUNDRY_TYPE_CONTEXTUAL,
                                  G_ADD_PRIVATE (FoundryDeviceProvider)
                                  G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static DexFuture *
foundry_device_provider_real_load (FoundryDeviceProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_device_provider_real_unload (FoundryDeviceProvider *self)
{
  return dex_future_new_true ();
}

static void
foundry_device_provider_finalize (GObject *object)
{
  FoundryDeviceProvider *self = (FoundryDeviceProvider *)object;
  FoundryDeviceProviderPrivate *priv = foundry_device_provider_get_instance_private (self);

  g_clear_object (&priv->store);

  G_OBJECT_CLASS (foundry_device_provider_parent_class)->finalize (object);
}

static void
foundry_device_provider_class_init (FoundryDeviceProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_device_provider_finalize;

  klass->load = foundry_device_provider_real_load;
  klass->unload = foundry_device_provider_real_unload;
}

static void
foundry_device_provider_init (FoundryDeviceProvider *self)
{
  FoundryDeviceProviderPrivate *priv = foundry_device_provider_get_instance_private (self);

  priv->store = g_list_store_new (FOUNDRY_TYPE_DEVICE);
  g_signal_connect_object (priv->store,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

void
foundry_device_provider_device_added (FoundryDeviceProvider *self,
                                      FoundryDevice         *device)
{
  FoundryDeviceProviderPrivate *priv = foundry_device_provider_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_DEVICE_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_DEVICE (device));

  _foundry_device_set_provider (device, self);

  g_list_store_append (priv->store, device);
}

void
foundry_device_provider_device_removed (FoundryDeviceProvider *self,
                                        FoundryDevice         *device)
{
  FoundryDeviceProviderPrivate *priv = foundry_device_provider_get_instance_private (self);
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_DEVICE_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_DEVICE (device));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (priv->store));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDevice) element = g_list_model_get_item (G_LIST_MODEL (priv->store), i);

      if (element == device)
        {
          g_list_store_remove (priv->store, i);
          _foundry_device_set_provider (device, NULL);
          return;
        }
    }

  g_critical ("%s did not contain device %s at %p",
              G_OBJECT_TYPE_NAME (self),
              G_OBJECT_TYPE_NAME (device),
              device);
}

static GType
foundry_device_provider_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DEVICE;
}

static guint
foundry_device_provider_get_n_items (GListModel *model)
{
  FoundryDeviceProvider *self = FOUNDRY_DEVICE_PROVIDER (model);
  FoundryDeviceProviderPrivate *priv = foundry_device_provider_get_instance_private (self);

  return g_list_model_get_n_items (G_LIST_MODEL (priv->store));
}

static gpointer
foundry_device_provider_get_item (GListModel *model,
                                  guint       position)
{
  FoundryDeviceProvider *self = FOUNDRY_DEVICE_PROVIDER (model);
  FoundryDeviceProviderPrivate *priv = foundry_device_provider_get_instance_private (self);

  return g_list_model_get_item (G_LIST_MODEL (priv->store), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_device_provider_get_item_type;
  iface->get_n_items = foundry_device_provider_get_n_items;
  iface->get_item = foundry_device_provider_get_item;
}

DexFuture *
foundry_device_provider_load (FoundryDeviceProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE_PROVIDER (self), NULL);

  return FOUNDRY_DEVICE_PROVIDER_GET_CLASS (self)->load (self);
}

DexFuture *
foundry_device_provider_unload (FoundryDeviceProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE_PROVIDER (self), NULL);

  return FOUNDRY_DEVICE_PROVIDER_GET_CLASS (self)->unload (self);
}

/**
 * foundry_device_provider_dup_name:
 * @self: a #FoundryDeviceProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Flatpak".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_device_provider_dup_name (FoundryDeviceProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DEVICE_PROVIDER (self), NULL);

  if (FOUNDRY_DEVICE_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_DEVICE_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}
