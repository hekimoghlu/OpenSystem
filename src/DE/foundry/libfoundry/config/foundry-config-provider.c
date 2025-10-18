/* foundry-config-provider.c
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

#include "foundry-config-provider-private.h"
#include "foundry-config-private.h"

typedef struct
{
  GListStore *store;
} FoundryConfigProviderPrivate;

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundryConfigProvider, foundry_config_provider, FOUNDRY_TYPE_CONTEXTUAL,
                                  G_ADD_PRIVATE (FoundryConfigProvider)
                                  G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static DexFuture *
foundry_config_provider_real_load (FoundryConfigProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_config_provider_real_unload (FoundryConfigProvider *self)
{
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);

  g_assert (FOUNDRY_IS_CONFIG_PROVIDER (self));

  g_list_store_remove_all (priv->store);

  return dex_future_new_true ();
}

static DexFuture *
foundry_config_provider_real_save (FoundryConfigProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_config_provider_real_copy (FoundryConfigProvider *self,
                                   FoundryConfig         *config)
{
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Operation not supported");
}

static DexFuture *
foundry_config_provider_real_delete (FoundryConfigProvider *self,
                                     FoundryConfig         *config)
{
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Operation not supported");
}

static void
foundry_config_provider_finalize (GObject *object)
{
  FoundryConfigProvider *self = (FoundryConfigProvider *)object;
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);

  g_clear_object (&priv->store);

  G_OBJECT_CLASS (foundry_config_provider_parent_class)->finalize (object);
}

static void
foundry_config_provider_class_init (FoundryConfigProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_config_provider_finalize;

  klass->load = foundry_config_provider_real_load;
  klass->unload = foundry_config_provider_real_unload;
  klass->save = foundry_config_provider_real_save;
  klass->copy = foundry_config_provider_real_copy;
  klass->delete = foundry_config_provider_real_delete;
}

static void
foundry_config_provider_init (FoundryConfigProvider *self)
{
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);

  priv->store = g_list_store_new (FOUNDRY_TYPE_CONFIG);
  g_signal_connect_object (priv->store,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

void
foundry_config_provider_config_added (FoundryConfigProvider *self,
                                      FoundryConfig         *config)
{
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_CONFIG (config));

  _foundry_config_set_provider (config, self);

  g_list_store_append (priv->store, config);
}

void
foundry_config_provider_config_removed (FoundryConfigProvider *self,
                                        FoundryConfig         *config)
{
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_CONFIG (config));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (priv->store));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryConfig) element = g_list_model_get_item (G_LIST_MODEL (priv->store), i);

      if (element == config)
        {
          g_list_store_remove (priv->store, i);
          _foundry_config_set_provider (config, NULL);
          return;
        }
    }

  g_critical ("%s did not contain config %s at %p",
              G_OBJECT_TYPE_NAME (self),
              G_OBJECT_TYPE_NAME (config),
              config);
}

static GType
foundry_config_provider_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_CONFIG;
}

static guint
foundry_config_provider_get_n_items (GListModel *model)
{
  FoundryConfigProvider *self = FOUNDRY_CONFIG_PROVIDER (model);
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);

  return g_list_model_get_n_items (G_LIST_MODEL (priv->store));
}

static gpointer
foundry_config_provider_get_item (GListModel *model,
                                  guint       position)
{
  FoundryConfigProvider *self = FOUNDRY_CONFIG_PROVIDER (model);
  FoundryConfigProviderPrivate *priv = foundry_config_provider_get_instance_private (self);

  return g_list_model_get_item (G_LIST_MODEL (priv->store), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_config_provider_get_item_type;
  iface->get_n_items = foundry_config_provider_get_n_items;
  iface->get_item = foundry_config_provider_get_item;
}

DexFuture *
foundry_config_provider_load (FoundryConfigProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self), NULL);

  return FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->load (self);
}

DexFuture *
foundry_config_provider_unload (FoundryConfigProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self), NULL);

  return FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->unload (self);
}

DexFuture *
foundry_config_provider_save (FoundryConfigProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self), NULL);

  return FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->save (self);
}

DexFuture *
foundry_config_provider_delete (FoundryConfigProvider *self,
                                FoundryConfig         *config)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (config), NULL);

  return FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->delete (self, config);
}

DexFuture *
foundry_config_provider_copy (FoundryConfigProvider *self,
                              FoundryConfig         *config)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (config), NULL);

  return FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->copy (self, config);
}

/**
 * foundry_config_provider_dup_name:
 * @self: a #FoundryConfigProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Flatpak".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_config_provider_dup_name (FoundryConfigProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONFIG_PROVIDER (self), NULL);

  if (FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_CONFIG_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}
