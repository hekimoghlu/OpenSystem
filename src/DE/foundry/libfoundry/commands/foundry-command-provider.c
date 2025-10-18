/* foundry-command-provider.c
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

#include "foundry-command-provider-private.h"
#include "foundry-command-private.h"

typedef struct
{
  GListStore *store;
} FoundryCommandProviderPrivate;

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundryCommandProvider, foundry_command_provider, FOUNDRY_TYPE_CONTEXTUAL,
                                  G_ADD_PRIVATE (FoundryCommandProvider)
                                  G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static DexFuture *
foundry_command_provider_real_load (FoundryCommandProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_command_provider_real_unload (FoundryCommandProvider *self)
{
  return dex_future_new_true ();
}

static void
foundry_command_provider_finalize (GObject *object)
{
  FoundryCommandProvider *self = (FoundryCommandProvider *)object;
  FoundryCommandProviderPrivate *priv = foundry_command_provider_get_instance_private (self);

  g_clear_object (&priv->store);

  G_OBJECT_CLASS (foundry_command_provider_parent_class)->finalize (object);
}

static void
foundry_command_provider_class_init (FoundryCommandProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_command_provider_finalize;

  klass->load = foundry_command_provider_real_load;
  klass->unload = foundry_command_provider_real_unload;
}

static void
foundry_command_provider_init (FoundryCommandProvider *self)
{
  FoundryCommandProviderPrivate *priv = foundry_command_provider_get_instance_private (self);

  priv->store = g_list_store_new (FOUNDRY_TYPE_COMMAND);
  g_signal_connect_object (priv->store,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

void
foundry_command_provider_command_added (FoundryCommandProvider *self,
                                        FoundryCommand         *command)
{
  FoundryCommandProviderPrivate *priv = foundry_command_provider_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_COMMAND_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_COMMAND (command));

  _foundry_command_set_provider (command, self);

  g_list_store_append (priv->store, command);
}

void
foundry_command_provider_command_removed (FoundryCommandProvider *self,
                                          FoundryCommand         *command)
{
  FoundryCommandProviderPrivate *priv = foundry_command_provider_get_instance_private (self);
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_COMMAND_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_COMMAND (command));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (priv->store));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryCommand) element = g_list_model_get_item (G_LIST_MODEL (priv->store), i);

      if (element == command)
        {
          g_list_store_remove (priv->store, i);
          _foundry_command_set_provider (command, NULL);
          return;
        }
    }

  g_critical ("%s did not contain command %s at %p",
              G_OBJECT_TYPE_NAME (self),
              G_OBJECT_TYPE_NAME (command),
              command);
}

static GType
foundry_command_provider_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_COMMAND;
}

static guint
foundry_command_provider_get_n_items (GListModel *model)
{
  FoundryCommandProvider *self = FOUNDRY_COMMAND_PROVIDER (model);
  FoundryCommandProviderPrivate *priv = foundry_command_provider_get_instance_private (self);

  return g_list_model_get_n_items (G_LIST_MODEL (priv->store));
}

static gpointer
foundry_command_provider_get_item (GListModel *model,
                                  guint       position)
{
  FoundryCommandProvider *self = FOUNDRY_COMMAND_PROVIDER (model);
  FoundryCommandProviderPrivate *priv = foundry_command_provider_get_instance_private (self);

  return g_list_model_get_item (G_LIST_MODEL (priv->store), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_command_provider_get_item_type;
  iface->get_n_items = foundry_command_provider_get_n_items;
  iface->get_item = foundry_command_provider_get_item;
}

DexFuture *
foundry_command_provider_load (FoundryCommandProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_PROVIDER (self), NULL);

  return FOUNDRY_COMMAND_PROVIDER_GET_CLASS (self)->load (self);
}

DexFuture *
foundry_command_provider_unload (FoundryCommandProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_PROVIDER (self), NULL);

  return FOUNDRY_COMMAND_PROVIDER_GET_CLASS (self)->unload (self);
}

/**
 * foundry_command_provider_dup_name:
 * @self: a #FoundryCommandProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Flatpak".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_command_provider_dup_name (FoundryCommandProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_COMMAND_PROVIDER (self), NULL);

  if (FOUNDRY_COMMAND_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_COMMAND_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}

