/* foundry-documentation-manager.c
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

#include "foundry-config.h"
#include "foundry-documentation.h"
#include "foundry-documentation-manager.h"
#include "foundry-documentation-matches-private.h"
#include "foundry-documentation-provider-private.h"
#include "foundry-documentation-provider.h"
#include "foundry-documentation-query.h"
#include "foundry-model-manager.h"
#include "foundry-inhibitor.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryDocumentationManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
  GListModel       *roots;
  DexFuture        *indexer;
  guint             indexing;
};

struct _FoundryDocumentationManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDocumentationManager, foundry_documentation_manager, FOUNDRY_TYPE_SERVICE)

enum {
  PROP_0,
  PROP_INDEXING,
  N_PROPS
};

enum {
  CHANGED,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

static void
foundry_documentation_manager_provider_added (PeasExtensionSet *set,
                                              PeasPluginInfo   *plugin_info,
                                              GObject          *addin,
                                              gpointer          user_data)
{
  FoundryDocumentationManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DOCUMENTATION_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));

  g_debug ("Adding FoundryDocumentationProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_documentation_provider_load (FOUNDRY_DOCUMENTATION_PROVIDER (addin)));
}

static void
foundry_documentation_manager_provider_removed (PeasExtensionSet *set,
                                                PeasPluginInfo   *plugin_info,
                                                GObject          *addin,
                                                gpointer          user_data)
{
  FoundryDocumentationManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DOCUMENTATION_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));

  g_debug ("Removing FoundryDocumentationProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_documentation_provider_unload (FOUNDRY_DOCUMENTATION_PROVIDER (addin)));
}

static DexFuture *
foundry_documentation_manager_index_fiber (gpointer data)
{
  FoundryDocumentationManager *self = data;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(DexFuture) future = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  dex_return_error_if_fail (PEAS_IS_EXTENSION_SET (self->addins));
  dex_return_error_if_fail (G_IS_LIST_MODEL (self->roots));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  if (n_items == 0)
    return dex_future_new_true ();

  if (++self->indexing == 1)
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INDEXING]);

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_ptr_array_add (futures, foundry_documentation_provider_index (provider, self->roots));
    }

  future = foundry_future_all (futures);
  dex_await (dex_ref (future), NULL);

  if (--self->indexing == 0)
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INDEXING]);

  return g_steal_pointer (&future);
}

/**
 * foundry_documentation_manager_index:
 * @self: a [class@Foundry.DocumentationManager]
 *
 * Ensures the documentation manager has indexed.
 *
 * This happens on demand but can be forced to start earlier by
 * applications which may want to delay operations until this
 * part has completed. Such an example is to now show a window
 * until the initial indexing has completed.
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_documentation_manager_index (FoundryDocumentationManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));

  if (self->indexer == NULL)
    {
      self->indexer = dex_scheduler_spawn (NULL, 0,
                                           foundry_documentation_manager_index_fiber,
                                           g_object_ref (self),
                                           g_object_unref);
      dex_future_disown (dex_ref (self->indexer));
    }

  return dex_ref (self->indexer);
}

static void
foundry_documentation_manager_roots_changed_cb (FoundryDocumentationManager *self,
                                                guint                        position,
                                                guint                        removed,
                                                guint                        added,
                                                GListModel                  *model)
{
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  g_assert (G_IS_LIST_MODEL (model));

  dex_clear (&self->indexer);

  g_signal_emit (self, signals[CHANGED], 0);
}

static DexFuture *
foundry_documentation_manager_start_fiber (gpointer user_data)
{
  FoundryDocumentationManager *self = user_data;
  g_autoptr(FoundryDocumentation) documentation = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListModel) flatten_roots = NULL;
  g_autoptr(GListStore) all_roots = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autofree char *documentation_id = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_documentation_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_documentation_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  /* First request that all of the providers pass the load phase */
  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_documentation_provider_load (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  /* Now collect all of the roots from various providers */
  all_roots = g_list_store_new (G_TYPE_LIST_MODEL);
  flatten_roots = foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (all_roots)));
  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_autoptr(GListModel) roots = foundry_documentation_provider_list_roots (provider);

      g_list_store_append (all_roots, roots);
    }

  g_signal_connect_object (flatten_roots,
                           "items-changed",
                           G_CALLBACK (foundry_documentation_manager_roots_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_set_object (&self->roots, flatten_roots);

  return dex_future_new_true ();
}

static DexFuture *
foundry_documentation_manager_start (FoundryService *service)
{
  FoundryDocumentationManager *self = (FoundryDocumentationManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_documentation_manager_start_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_documentation_manager_stop (FoundryService *service)
{
  FoundryDocumentationManager *self = (FoundryDocumentationManager *)service;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  g_clear_object (&self->roots);

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_documentation_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_documentation_manager_provider_removed),
                                        self);

  dex_clear (&self->indexer);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_documentation_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_documentation_manager_constructed (GObject *object)
{
  FoundryDocumentationManager *self = (FoundryDocumentationManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_documentation_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_DOCUMENTATION_PROVIDER,
                                         "context", context,
                                         NULL);
}

static void
foundry_documentation_manager_finalize (GObject *object)
{
  FoundryDocumentationManager *self = (FoundryDocumentationManager *)object;

  g_clear_object (&self->addins);
  g_clear_object (&self->roots);

  G_OBJECT_CLASS (foundry_documentation_manager_parent_class)->finalize (object);
}

static void
foundry_documentation_manager_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundryDocumentationManager *self = FOUNDRY_DOCUMENTATION_MANAGER (object);

  switch (prop_id)
    {
    case PROP_INDEXING:
      g_value_set_boolean (value, foundry_documentation_manager_is_indexing (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_manager_class_init (FoundryDocumentationManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_documentation_manager_constructed;
  object_class->finalize = foundry_documentation_manager_finalize;
  object_class->get_property = foundry_documentation_manager_get_property;

  service_class->start = foundry_documentation_manager_start;
  service_class->stop = foundry_documentation_manager_stop;

  /**
   * FoundryDocumentationManager:indexing: (getter is_indexing)
   *
   * If the documentation is currently indexing documentation.
   */
  properties[PROP_INDEXING] =
    g_param_spec_boolean ("indexing", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  /**
   * FoundryDocumentationManager:changed:
   *
   * The "changed" signal is emitted when the manager discovers that
   * there is documentation which has changed and needs to be re-indexed.
   *
   * Applications that care about this can respond by calling
   * [method@Foundry.DocumentationManager.index] to re-index the
   * documentation at their point of convenience.
   */
  signals[CHANGED] =
    g_signal_new ("changed",
                 G_TYPE_FROM_CLASS (klass),
                 G_SIGNAL_RUN_LAST,
                 0,
                 NULL, NULL,
                 NULL,
                 G_TYPE_NONE, 0);
}

static void
foundry_documentation_manager_init (FoundryDocumentationManager *self)
{
}

static DexFuture *
foundry_documentation_manager_query_fiber (FoundryDocumentationManager *self,
                                           FoundryDocumentationQuery   *query)
{
  g_autoptr(FoundryDocumentationMatches) matches = NULL;
  g_autoptr(GError) error = NULL;
  DexFuture *everything = NULL;

  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION_QUERY (query));

  /* Wait for startup to complete */
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (self)), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* Wait for indexing to complete */
  if (!dex_await (foundry_documentation_manager_index (self), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  matches = foundry_documentation_matches_new (query);

  /* Query providers and await the creation of the immediate
   * GListModel. If it has a future associated with it, then
   * swap out our future for the future that will complete when
   * the model has completed so we can provide the same feature
   * to our caller (a future list model with immediate results plus
   * ability to await on full result set).
   */
  if (self->addins != NULL)
    {
      g_autoptr(GListModel) addins = g_object_ref (G_LIST_MODEL (self->addins));
      g_autoptr(GPtrArray) futures = g_ptr_array_new_with_free_func (dex_unref);
      guint n_items = g_list_model_get_n_items (addins);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (addins, i);

          g_ptr_array_add (futures,
                           foundry_documentation_provider_query (provider, query, matches));
        }

      if (futures->len > 0)
        everything = foundry_future_all (futures);
    }

  foundry_documentation_matches_set_future (matches, everything);

  return dex_future_new_take_object (g_steal_pointer (&matches));
}

/**
 * foundry_documentation_manager_query:
 * @self: a [class@Foundry.DocumentationManager]
 * @query: a [class@Foundry.DocumentationQuery]
 *
 * Consumers can call [method@Foundry.DocumentationMatches.await] to get a
 * future that will complete when all matches have been populated.
 *
 * Returns: (transfer full) (not nullable): a [class@Dex.Future] that resolves
 *    to [Foundry.DocumentationMatches].
 */
DexFuture *
foundry_documentation_manager_query (FoundryDocumentationManager *self,
                                     FoundryDocumentationQuery   *query)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_documentation_manager_query_fiber),
                                  2,
                                  FOUNDRY_TYPE_DOCUMENTATION_MANAGER, self,
                                  FOUNDRY_TYPE_DOCUMENTATION_QUERY, query);
}

/**
 * foundry_documentation_manager_is_indexing:
 * @self: a [class@Foundry.DocumentationManager]
 *
 * If the documentation manager is currently indexing.
 */
gboolean
foundry_documentation_manager_is_indexing (FoundryDocumentationManager *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self), FALSE);

  return self->indexing > 0;
}

/**
 * foundry_documentation_manager_find_by_uri:
 * @self: a [class@Foundry.DocumentationManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.Documentation] or rejects with error.
 */
DexFuture *
foundry_documentation_manager_find_by_uri (FoundryDocumentationManager *self,
                                           const char                  *uri)
{
  g_autoptr(GPtrArray) futures = NULL;
  GListModel *model;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  dex_return_error_if_fail (uri != NULL);

  model = G_LIST_MODEL (self->addins);

  if (!(n_items = g_list_model_get_n_items (model)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Not found");

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (model, i);

      g_ptr_array_add (futures, foundry_documentation_provider_find_by_uri (provider, uri));
    }

  return dex_future_anyv ((DexFuture **)futures->pdata, futures->len);
}

static DexFuture *
foundry_documentation_manager_list_children_cb (DexFuture *completed,
                                                gpointer   user_data)
{
  g_autoptr(GListStore) store = NULL;
  guint size;

  g_assert (DEX_IS_FUTURE_SET (completed));

  size = dex_future_set_get_size (DEX_FUTURE_SET (completed));
  store = g_list_store_new (G_TYPE_LIST_MODEL);

  for (guint i = 0; i < size; i++)
    {
      const GValue *value;

      if ((value = dex_future_set_get_value_at (DEX_FUTURE_SET (completed), i, NULL)) &&
          G_VALUE_HOLDS (value, G_TYPE_LIST_MODEL))
        g_list_store_append (store, g_value_get_object (value));
    }

  return dex_future_new_take_object (foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store))));
}

static DexFuture *
foundry_documentation_manager_list_children_fiber (FoundryDocumentationManager *self,
                                                   FoundryDocumentation        *parent)
{
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;
  GListModel *model;
  guint n_items;

  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  g_assert (!parent || FOUNDRY_IS_DOCUMENTATION (parent));

  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (self)), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  dex_await (foundry_documentation_manager_index (self), NULL);

  model = G_LIST_MODEL (self->addins);

  if (!(n_items = g_list_model_get_n_items (model)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Not found");


  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (model, i);

      g_ptr_array_add (futures, foundry_documentation_provider_list_children (provider, parent));
    }

  return dex_future_finally (dex_future_anyv ((DexFuture **)futures->pdata, futures->len),
                             foundry_documentation_manager_list_children_cb,
                             NULL, NULL);
}

/**
 * foundry_documentation_manager_list_children:
 * @self: a [class@Foundry.DocumentationManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] or rejects with error.
 */
DexFuture *
foundry_documentation_manager_list_children (FoundryDocumentationManager *self,
                                             FoundryDocumentation        *parent)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));
  dex_return_error_if_fail (!parent || FOUNDRY_IS_DOCUMENTATION (parent));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_documentation_manager_list_children_fiber),
                                  2,
                                  FOUNDRY_TYPE_DOCUMENTATION_MANAGER, self,
                                  FOUNDRY_TYPE_DOCUMENTATION, parent);
}

static DexFuture *
foundry_documentation_manager_list_bundles_cb (DexFuture *completed,
                                               gpointer   user_data)
{
  g_autoptr(GListStore) store = NULL;
  guint size;

  g_assert (DEX_IS_FUTURE_SET (completed));

  size = dex_future_set_get_size (DEX_FUTURE_SET (completed));
  store = g_list_store_new (G_TYPE_LIST_MODEL);

  for (guint i = 0; i < size; i++)
    {
      const GValue *value;

      if ((value = dex_future_set_get_value_at (DEX_FUTURE_SET (completed), i, NULL)) &&
          G_VALUE_HOLDS (value, G_TYPE_LIST_MODEL))
        g_list_store_append (store, g_value_get_object (value));

    }

  return dex_future_new_take_object (foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store))));
}

/**
 * foundry_documentation_manager_list_bundles:
 * @self: a [class@Foundry.DocumentationManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.DocumentationBundle]
 *   or rejects with error.
 */
DexFuture *
foundry_documentation_manager_list_bundles (FoundryDocumentationManager *self)
{
  g_autoptr(GPtrArray) futures = NULL;
  GListModel *model;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MANAGER (self));

  model = G_LIST_MODEL (self->addins);

  if (!(n_items = g_list_model_get_n_items (model)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Not found");

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDocumentationProvider) provider = g_list_model_get_item (model, i);

      g_ptr_array_add (futures, foundry_documentation_provider_list_bundles (provider));
    }

  return dex_future_finally (dex_future_anyv ((DexFuture **)futures->pdata, futures->len),
                             foundry_documentation_manager_list_bundles_cb,
                             NULL, NULL);
}
