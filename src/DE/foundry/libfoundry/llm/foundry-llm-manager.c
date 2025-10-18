/* foundry-llm-manager.c
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

#include "foundry-llm-manager.h"
#include "foundry-llm-model.h"
#include "foundry-llm-provider-private.h"
#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-model-manager.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

struct _FoundryLlmManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
  PeasExtensionSet *tools;
};

struct _FoundryLlmManagerClass
{
  FoundryServiceClass parent_class;
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryLlmManager, foundry_llm_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
foundry_llm_manager_provider_added (PeasExtensionSet *set,
                                    PeasPluginInfo   *plugin_info,
                                    GObject          *addin,
                                    gpointer          user_data)
{
  FoundryLlmManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_LLM_PROVIDER (addin));
  g_assert (FOUNDRY_IS_LLM_MANAGER (self));

  g_debug ("Adding FoundryLlmProvider of type `%s`", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_llm_provider_load (FOUNDRY_LLM_PROVIDER (addin)));
}

static void
foundry_llm_manager_provider_removed (PeasExtensionSet *set,
                                      PeasPluginInfo   *plugin_info,
                                      GObject          *addin,
                                      gpointer          user_data)
{
  FoundryLlmManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_LLM_PROVIDER (addin));
  g_assert (FOUNDRY_IS_LLM_MANAGER (self));

  g_debug ("Removing FoundryLlmProvider of type `%s`", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_llm_provider_unload (FOUNDRY_LLM_PROVIDER (addin)));
}

static DexFuture *
foundry_llm_manager_start_fiber (gpointer user_data)
{
  FoundryLlmManager *self = user_data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_LLM_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_llm_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_llm_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLlmProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_ptr_array_add (futures, foundry_llm_provider_load (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  return dex_future_new_true ();
}

static DexFuture *
foundry_llm_manager_start (FoundryService *service)
{
  FoundryLlmManager *self = (FoundryLlmManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_LLM_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_llm_manager_start_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_llm_manager_stop (FoundryService *service)
{
  FoundryLlmManager *self = (FoundryLlmManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_llm_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_llm_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLlmProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_ptr_array_add (futures, foundry_llm_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_llm_manager_constructed (GObject *object)
{
  FoundryLlmManager *self = (FoundryLlmManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_llm_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_LLM_PROVIDER,
                                         "context", context,
                                         NULL);

  g_signal_connect_object (self->addins,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);

  self->tools = peas_extension_set_new (NULL,
                                        FOUNDRY_TYPE_LLM_PROVIDER,
                                        "context", context,
                                        NULL);
}

static void
foundry_llm_manager_dispose (GObject *object)
{
  FoundryLlmManager *self = (FoundryLlmManager *)object;

  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_llm_manager_parent_class)->dispose (object);
}

static void
foundry_llm_manager_class_init (FoundryLlmManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_llm_manager_constructed;
  object_class->dispose = foundry_llm_manager_dispose;

  service_class->start = foundry_llm_manager_start;
  service_class->stop = foundry_llm_manager_stop;
}

static void
foundry_llm_manager_init (FoundryLlmManager *self)
{
}

static GType
foundry_llm_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_LLM_PROVIDER;
}

static guint
foundry_llm_manager_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (G_LIST_MODEL (FOUNDRY_LLM_MANAGER (model)->addins));
}

static gpointer
foundry_llm_manager_get_item (GListModel *model,
                              guint       position)
{
  return g_list_model_get_item (G_LIST_MODEL (FOUNDRY_LLM_MANAGER (model)->addins), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_llm_manager_get_item_type;
  iface->get_n_items = foundry_llm_manager_get_n_items;
  iface->get_item = foundry_llm_manager_get_item;
}

static DexFuture *
foundry_llm_manager_list_models_fiber (gpointer data)
{
  FoundryLlmManager *self = data;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GError) error = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_LLM_MANAGER (self));

  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (self)), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLlmProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_ptr_array_add (futures, foundry_llm_provider_list_models (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  store = g_list_store_new (G_TYPE_LIST_MODEL);

  for (guint i = 0; i < futures->len; i++)
    {
      DexFuture *future = g_ptr_array_index (futures, i);
      const GValue *value;

      if ((value = dex_future_get_value (future, NULL)) &&
          G_VALUE_HOLDS (value, G_TYPE_LIST_MODEL))
        g_list_store_append (store, g_value_get_object (value));
    }

  return dex_future_new_take_object (foundry_flatten_list_model_new (G_LIST_MODEL (g_steal_pointer (&store))));
}

/**
 * foundry_llm_manager_list_models:
 * @self: a [class@Foundry.LlmManager]
 *
 * List models from all providers.
 *
 * The resulting [iface@Gio.ListModel] is asynchronously populated.
 * If you want to be sure that all providers have completed populating,
 * you may await completion by calling [func@Foundry.list_model_await].
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.LlmModel].
 */
DexFuture *
foundry_llm_manager_list_models (FoundryLlmManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_MANAGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_llm_manager_list_models_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_llm_manager_find_model_cb (DexFuture *completed,
                                   gpointer   data)
{
  const char *name = data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GListModel) list = NULL;
  guint n_items;

  if (!(list = dex_await_object (dex_ref (completed), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  n_items = g_list_model_get_n_items (list);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLlmModel) model = g_list_model_get_item (list, i);
      g_autofree char *model_name = foundry_llm_model_dup_name (model);

      if (g_strcmp0 (model_name, name) == 0)
        return dex_future_new_take_object (g_steal_pointer (&model));
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

/**
 * foundry_llm_manager_find_model:
 * @self: a [class@Foundry.LlmManager]
 * @name: the name of the model
 *
 * Finds the first model which matches @name.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.LlmModel] or rejects with error.
 */
DexFuture *
foundry_llm_manager_find_model (FoundryLlmManager *self,
                                const char        *name)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_MANAGER (self));
  dex_return_error_if_fail (name != NULL);

  return dex_future_then (foundry_llm_manager_list_models (self),
                          foundry_llm_manager_find_model_cb,
                          g_strdup (name),
                          g_free);
}

/**
 * foundry_llm_manager_list_tools:
 * @self: a [class@Foundry.LlmManager]
 *
 * The resulting [iface@Gio.ListModel] is asynchronously populated.
 * If you want to be sure that all providers have completed populating,
 * you may await completion by calling [func@Foundry.list_model_await].
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.LlmTool].
 */
DexFuture *
foundry_llm_manager_list_tools (FoundryLlmManager *self)
{
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_LLM_MANAGER (self));

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLlmProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_ptr_array_add (futures, foundry_llm_provider_list_tools (provider));
    }

  return _foundry_flatten_list_model_new_from_futures (futures);
}
