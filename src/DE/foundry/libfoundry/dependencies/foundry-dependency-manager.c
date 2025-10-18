/* foundry-dependency-manager.c
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

#include <glib/gstdio.h>

#include <libpeas.h>

#include "foundry-build-manager.h"
#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-dependency.h"
#include "foundry-dependency-manager.h"
#include "foundry-dependency-provider-private.h"
#include "foundry-dependency-provider.h"
#include "foundry-inhibitor.h"
#include "foundry-model-manager.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryDependencyManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
};

struct _FoundryDependencyManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDependencyManager, foundry_dependency_manager, FOUNDRY_TYPE_SERVICE)

static void
foundry_dependency_manager_update_action (FoundryService *service,
                                          const char     *action_name,
                                          GVariant       *param)
{
  FoundryDependencyManager *self = (FoundryDependencyManager *)service;
  g_autoptr(FoundryConfigManager) config_manager = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  int pty_fd;

  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  build_manager = foundry_context_dup_build_manager (context);
  config_manager = foundry_context_dup_config_manager (context);
  config = foundry_config_manager_dup_config (config_manager);
  pty_fd = foundry_build_manager_get_default_pty (build_manager);

  foundry_dependency_manager_update_dependencies (self, config, pty_fd, NULL);
}

static void
foundry_dependency_manager_provider_added (PeasExtensionSet *set,
                                           PeasPluginInfo   *plugin_info,
                                           GObject          *addin,
                                           gpointer          user_data)
{
  FoundryDependencyManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DEPENDENCY_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (self));

  g_debug ("Adding FoundryDependencyProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_dependency_provider_load (FOUNDRY_DEPENDENCY_PROVIDER (addin)));
}

static void
foundry_dependency_manager_provider_removed (PeasExtensionSet *set,
                                             PeasPluginInfo   *plugin_info,
                                             GObject          *addin,
                                             gpointer          user_data)
{
  FoundryDependencyManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DEPENDENCY_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (self));

  g_debug ("Removing FoundryDependencyProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_dependency_provider_unload (FOUNDRY_DEPENDENCY_PROVIDER (addin)));
}

static DexFuture *
foundry_dependency_manager_start_fiber (gpointer user_data)
{
  FoundryDependencyManager *self = user_data;
  g_autoptr(FoundryDependency) dependency = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autofree char *dependency_id = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_dependency_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_dependency_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDependencyProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_dependency_provider_load (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  return dex_future_new_true ();
}

static DexFuture *
foundry_dependency_manager_start (FoundryService *service)
{
  FoundryDependencyManager *self = (FoundryDependencyManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_dependency_manager_start_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_dependency_manager_stop (FoundryService *service)
{
  FoundryDependencyManager *self = (FoundryDependencyManager *)service;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_dependency_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_dependency_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDependencyProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_dependency_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_dependency_manager_constructed (GObject *object)
{
  FoundryDependencyManager *self = (FoundryDependencyManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_dependency_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_DEPENDENCY_PROVIDER,
                                         "context", context,
                                         NULL);
}

static void
foundry_dependency_manager_finalize (GObject *object)
{
  FoundryDependencyManager *self = (FoundryDependencyManager *)object;

  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_dependency_manager_parent_class)->finalize (object);
}

static void
foundry_dependency_manager_class_init (FoundryDependencyManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_dependency_manager_constructed;
  object_class->finalize = foundry_dependency_manager_finalize;

  service_class->start = foundry_dependency_manager_start;
  service_class->stop = foundry_dependency_manager_stop;

  foundry_service_class_set_action_prefix (service_class, "dependency-manager");
  foundry_service_class_install_action (service_class, "update", NULL, foundry_dependency_manager_update_action);
}

static void
foundry_dependency_manager_init (FoundryDependencyManager *self)
{
}

static DexFuture *
add_to_list_store (DexFuture *completed,
                   gpointer   user_data)
{
  GListStore *store = user_data;
  g_autoptr(GListModel) model = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (G_IS_LIST_STORE (store));

  model = dex_await_object (dex_ref (completed), NULL);
  g_assert (G_IS_LIST_MODEL (model));

  g_list_store_append (store, model);

  return foundry_list_model_await (model);
}

static DexFuture *
foundry_dependency_manager_list_dependencies_fiber (gpointer data)
{
  FoundryPair *pair = data;
  FoundryDependencyManager *self = FOUNDRY_DEPENDENCY_MANAGER (pair->first);
  FoundryConfig *config = FOUNDRY_CONFIG (pair->second);
  g_autoptr(GListModel) flatten = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;
  DexFuture *future;
  guint n_items;

  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (self));
  g_assert (FOUNDRY_IS_CONFIG (config));

  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (self)), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  dex_return_error_if_fail (self->addins != NULL);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);
  store = g_list_store_new (G_TYPE_LIST_MODEL);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDependencyProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       dex_future_then (foundry_dependency_provider_list_dependencies (provider, config, NULL),
                                        add_to_list_store,
                                        g_object_ref (store),
                                        g_object_unref));
    }

  if (futures->len > 0)
    future = foundry_future_all (futures);
  else
    future = dex_future_new_true ();

  flatten = foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store)));
  foundry_list_model_set_future (flatten, future);

  return dex_future_new_take_object (g_steal_pointer (&flatten));
}


/**
 * foundry_dependency_manager_list_dependencies:
 * @self: a [class@Foundry.DependencyManager]
 * @config: a [class@Foundry.Config]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel].
 */
DexFuture *
foundry_dependency_manager_list_dependencies (FoundryDependencyManager *self,
                                              FoundryConfig            *config)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEPENDENCY_MANAGER (self));
  dex_return_error_if_fail (FOUNDRY_IS_CONFIG (config));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_dependency_manager_list_dependencies_fiber,
                              foundry_pair_new (self, config),
                              (GDestroyNotify) foundry_pair_free);
}

typedef struct
{
  FoundryDependencyManager *self;
  DexCancellable *cancellable;
  FoundryConfig *config;
  int pty_fd;
} UpdateDependencies;

static void
update_dependencies_free (UpdateDependencies *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->config);
  dex_clear (&state->cancellable);
  g_clear_fd (&state->pty_fd, NULL);
  g_free (state);
}

static DexFuture *
foundry_dependency_manager_update_dependencies_fiber (gpointer data)
{
  UpdateDependencies *state = data;
  g_autoptr(GPtrArray) providers = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_DEPENDENCY_MANAGER (state->self));
  g_assert (state->pty_fd >= -1);
  g_assert (!state->cancellable || DEX_IS_CANCELLABLE (state->cancellable));

  if (state->self->addins == NULL)
    return dex_future_new_true ();

  /* Copy providers in case they change while we await */
  providers = g_ptr_array_new_with_free_func (g_object_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (state->self->addins));
  for (guint i = 0; i < n_items; i++)
    g_ptr_array_add (providers, g_list_model_get_item (G_LIST_MODEL (state->self->addins), i));

  for (guint i = 0; i < providers->len; i++)
    {
      FoundryDependencyProvider *provider = g_ptr_array_index (providers, i);
      g_autoptr(GListModel) dependencies = NULL;

      if ((dependencies = dex_await_object (foundry_dependency_provider_list_dependencies (provider, state->config, NULL), NULL)))
        dex_await (foundry_list_model_await (dependencies), NULL);

      dex_await (foundry_dependency_provider_update_dependencies (provider, state->config, dependencies, state->pty_fd, state->cancellable), NULL);
    }

  return dex_future_new_true ();
}

/**
 * foundry_dependency_manager_update_dependencies:
 * @self: a [class@Foundry.DependencyManager]
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_dependency_manager_update_dependencies (FoundryDependencyManager *self,
                                                FoundryConfig            *config,
                                                int                       pty_fd,
                                                DexCancellable           *cancellable)
{
  UpdateDependencies *state;

  dex_return_error_if_fail (FOUNDRY_IS_DEPENDENCY_MANAGER (self));
  dex_return_error_if_fail (FOUNDRY_IS_CONFIG (config));
  dex_return_error_if_fail (pty_fd >= -1);
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  state = g_new0 (UpdateDependencies, 1);
  state->self = g_object_ref (self);
  state->config = g_object_ref (config);
  state->pty_fd = dup (pty_fd);
  state->cancellable = cancellable ? dex_ref (cancellable) : dex_cancellable_new ();

  return dex_scheduler_spawn (NULL, 0,
                              foundry_dependency_manager_update_dependencies_fiber,
                              state,
                              (GDestroyNotify) update_dependencies_free);
}
