/* foundry-debugger-manager.c
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

#include "foundry-build-pipeline.h"
#include "foundry-command.h"
#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-debugger-manager.h"
#include "foundry-debugger-provider-private.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

struct _FoundryDebuggerManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
};

struct _FoundryDebuggerManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDebuggerManager, foundry_debugger_manager, FOUNDRY_TYPE_SERVICE)

static void
foundry_debugger_manager_provider_added (PeasExtensionSet *set,
                                         PeasPluginInfo   *plugin_info,
                                         GObject          *addin,
                                         gpointer          user_data)
{
  FoundryDebuggerManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DEBUGGER_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DEBUGGER_MANAGER (self));

  g_debug ("Adding FoundryDebugger of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_debugger_provider_load (FOUNDRY_DEBUGGER_PROVIDER (addin)));
}

static void
foundry_debugger_manager_provider_removed (PeasExtensionSet *set,
                                           PeasPluginInfo   *plugin_info,
                                           GObject          *addin,
                                           gpointer          user_data)
{
  FoundryDebuggerManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DEBUGGER_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DEBUGGER_MANAGER (self));

  g_debug ("Removing FoundryDebugger of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_debugger_provider_unload (FOUNDRY_DEBUGGER_PROVIDER (addin)));
}

static DexFuture *
foundry_debugger_manager_start (FoundryService *service)
{
  FoundryDebuggerManager *self = (FoundryDebuggerManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_debugger_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_debugger_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDebuggerProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_debugger_provider_load (provider));
    }

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static DexFuture *
foundry_debugger_manager_stop (FoundryService *service)
{
  FoundryDebuggerManager *self = (FoundryDebuggerManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_debugger_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_debugger_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDebuggerProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_debugger_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_debugger_manager_constructed (GObject *object)
{
  FoundryDebuggerManager *self = (FoundryDebuggerManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_debugger_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_DEBUGGER_PROVIDER,
                                         "context", context,
                                         NULL);
}

static void
foundry_debugger_manager_finalize (GObject *object)
{
  FoundryDebuggerManager *self = (FoundryDebuggerManager *)object;

  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_debugger_manager_parent_class)->finalize (object);
}

static void
foundry_debugger_manager_class_init (FoundryDebuggerManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_debugger_manager_constructed;
  object_class->finalize = foundry_debugger_manager_finalize;

  service_class->start = foundry_debugger_manager_start;
  service_class->stop = foundry_debugger_manager_stop;
}

static void
foundry_debugger_manager_init (FoundryDebuggerManager *self)
{
}

static DexFuture *
foundry_debugger_manager_discover_fiber (FoundryDebuggerManager *self,
                                         FoundryBuildPipeline   *pipeline,
                                         FoundryCommand         *command)
{
  g_autoptr(FoundryDebuggerProvider) best = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autofree char *preferred = NULL;
  guint n_items = 0;
  int best_priority = G_MININT;

  g_assert (FOUNDRY_IS_DEBUGGER_MANAGER (self));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_COMMAND (command));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  settings = foundry_context_load_settings (context, "app.devsuite.foundry.run", NULL);
  preferred = foundry_settings_get_string (settings, "preferred-debugger");

  if (self->addins != NULL)
    n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDebuggerProvider) provider = NULL;
      g_autoptr(PeasPluginInfo) info = NULL;
      g_autoptr(GError) error = NULL;
      int priority;

      provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      priority = dex_await_int (foundry_debugger_provider_supports (provider, pipeline, command), &error);

      if (error != NULL)
        continue;

      /* We allow the user or project to set their preferred debugger which
       * will override all the internal self-guessing by plugins.
       */
      if (!foundry_str_empty0 (preferred) &&
          (info = foundry_debugger_provider_dup_plugin_info (provider)) &&
          foundry_str_equal0 (preferred, peas_plugin_info_get_module_name (info)))
        {
          g_set_object (&best, provider);
          best_priority = G_MAXINT;
          continue;
        }

      if (priority > best_priority || best == NULL)
        {
          g_set_object (&best, provider);
          best_priority = priority;
        }
    }

  if (best != NULL)
    return dex_future_new_take_object (g_steal_pointer (&best));

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

/**
 * foundry_debugger_manager_discover:
 * @self: a [class@Foundry.DebuggerManager]
 * @pipeline: (nullable): a [class@Foundry.BuildPipeline]
 * @command: a [class@Foundry.Command]
 *
 * Discovers a [class@Foundry.DebuggerProvider] that is likely to be
 * usable with the pipeline and command.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to an
 *   [class@Foundry.DebuggerProvider].
 */
DexFuture *
foundry_debugger_manager_discover (FoundryDebuggerManager *self,
                                   FoundryBuildPipeline   *pipeline,
                                   FoundryCommand         *command)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MANAGER (self), NULL);
  g_return_val_if_fail (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline), NULL);
  g_return_val_if_fail (FOUNDRY_IS_COMMAND (command), NULL);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_debugger_manager_discover_fiber),
                                  3,
                                  FOUNDRY_TYPE_DEBUGGER_MANAGER, self,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline,
                                  FOUNDRY_TYPE_COMMAND, command);
}

/**
 * foundry_debugger_manager_find:
 * @self: a [class@Foundry.DebuggerManager]
 * @module_name: module name of the plugin
 *
 * Returns: (transfer full) (nullable): a [class@Foundry.DebuggerProvider]
 *
 * Since: 1.1
 */
FoundryDebuggerProvider *
foundry_debugger_manager_find (FoundryDebuggerManager *self,
                               const char             *module_name)
{
  guint n_items = 0;

  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MANAGER (self), NULL);
  g_return_val_if_fail (module_name != NULL, NULL);

  if (self->addins != NULL)
    n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDebuggerProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_autoptr(PeasPluginInfo) info = foundry_debugger_provider_dup_plugin_info (provider);

      if (g_strcmp0 (module_name, peas_plugin_info_get_module_name (info)) == 0)
        return g_steal_pointer (&provider);
    }

  return NULL;
}
