/* foundry-command-manager.c
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

#include "foundry-command.h"
#include "foundry-command-manager.h"
#include "foundry-command-provider-private.h"
#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-model-manager.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

struct _FoundryCommandManager
{
  FoundryService    parent_instance;
  GListModel       *flatten;
  PeasExtensionSet *addins;
};

struct _FoundryCommandManagerClass
{
  FoundryServiceClass parent_class;
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryCommandManager, foundry_command_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
foundry_command_manager_provider_added (PeasExtensionSet *set,
                                        PeasPluginInfo   *plugin_info,
                                        GObject          *addin,
                                        gpointer          user_data)
{
  FoundryCommandManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_COMMAND_PROVIDER (addin));
  g_assert (FOUNDRY_IS_COMMAND_MANAGER (self));

  g_debug ("Adding FoundryCommandProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_command_provider_load (FOUNDRY_COMMAND_PROVIDER (addin)));
}

static void
foundry_command_manager_provider_removed (PeasExtensionSet *set,
                                          PeasPluginInfo   *plugin_info,
                                          GObject          *addin,
                                          gpointer          user_data)
{
  FoundryCommandManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_COMMAND_PROVIDER (addin));
  g_assert (FOUNDRY_IS_COMMAND_MANAGER (self));

  g_debug ("Removing FoundryCommandProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_command_provider_unload (FOUNDRY_COMMAND_PROVIDER (addin)));
}

static DexFuture *
foundry_command_manager_start_fiber (gpointer user_data)
{
  FoundryCommandManager *self = user_data;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_COMMAND_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_command_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_command_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryCommandProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_command_provider_load (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  return dex_future_new_true ();
}

static DexFuture *
foundry_command_manager_start (FoundryService *service)
{
  FoundryCommandManager *self = (FoundryCommandManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_COMMAND_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_command_manager_start_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_command_manager_stop (FoundryService *service)
{
  FoundryCommandManager *self = (FoundryCommandManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_command_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_command_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryCommandProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_command_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_command_manager_constructed (GObject *object)
{
  FoundryCommandManager *self = (FoundryCommandManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_command_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_COMMAND_PROVIDER,
                                         "context", context,
                                         NULL);

  g_object_set (self->flatten,
                "model", self->addins,
                NULL);
}

static void
foundry_command_manager_finalize (GObject *object)
{
  FoundryCommandManager *self = (FoundryCommandManager *)object;

  g_clear_object (&self->flatten);
  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_command_manager_parent_class)->finalize (object);
}

static void
foundry_command_manager_class_init (FoundryCommandManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_command_manager_constructed;
  object_class->finalize = foundry_command_manager_finalize;

  service_class->start = foundry_command_manager_start;
  service_class->stop = foundry_command_manager_stop;
}

static void
foundry_command_manager_init (FoundryCommandManager *self)
{
  self->flatten = foundry_flatten_list_model_new (NULL);

  g_signal_connect_object (self->flatten,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

static GType
foundry_command_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_COMMAND;
}

static guint
foundry_command_manager_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (FOUNDRY_COMMAND_MANAGER (model)->flatten);
}

static gpointer
foundry_command_manager_get_item (GListModel *model,
                                  guint       position)
{
  return g_list_model_get_item (FOUNDRY_COMMAND_MANAGER (model)->flatten, position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_command_manager_get_item_type;
  iface->get_n_items = foundry_command_manager_get_n_items;
  iface->get_item = foundry_command_manager_get_item;
}

/**
 * foundry_command_manager_find_command:
 * @self: a #FoundryCommandManager
 * @command_id: an identifier matching a #FoundryCommand:id
 *
 * Looks through available commands to find one matching @command_id.
 *
 * Returns: (transfer full) (nullable): a #FoundryCommand or %NULL
 */
FoundryCommand *
foundry_command_manager_find_command (FoundryCommandManager *self,
                                      const char            *command_id)
{
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_COMMAND_MANAGER (self), NULL);
  g_return_val_if_fail (command_id != NULL, NULL);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryCommand) command = g_list_model_get_item (G_LIST_MODEL (self), i);
      g_autofree char *id = foundry_command_dup_id (command);

      if (g_strcmp0 (command_id, id) == 0)
        return g_steal_pointer (&command);
    }

  return NULL;
}

