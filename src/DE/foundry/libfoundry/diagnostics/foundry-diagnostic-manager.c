/* foundry-diagnostic-manager.c
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
#include "foundry-diagnostic-manager.h"
#include "foundry-diagnostic-provider-private.h"
#include "foundry-diagnostic.h"
#include "foundry-file-manager.h"
#include "foundry-inhibitor.h"
#include "foundry-model-manager.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryDiagnosticManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
};

struct _FoundryDiagnosticManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDiagnosticManager, foundry_diagnostic_manager, FOUNDRY_TYPE_SERVICE)

static void
foundry_diagnostic_manager_provider_added (PeasExtensionSet *set,
                                           PeasPluginInfo   *plugin_info,
                                           GObject          *addin,
                                           gpointer          user_data)
{
  FoundryDiagnosticManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));

  g_debug ("Adding FoundryDiagnosticProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_diagnostic_provider_load (FOUNDRY_DIAGNOSTIC_PROVIDER (addin)));
}

static void
foundry_diagnostic_manager_provider_removed (PeasExtensionSet *set,
                                             PeasPluginInfo   *plugin_info,
                                             GObject          *addin,
                                             gpointer          user_data)
{
  FoundryDiagnosticManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (addin));
  g_assert (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));

  g_debug ("Removing FoundryDiagnosticProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_diagnostic_provider_unload (FOUNDRY_DIAGNOSTIC_PROVIDER (addin)));
}

static DexFuture *
foundry_diagnostic_manager_start (FoundryService *service)
{
  FoundryDiagnosticManager *self = (FoundryDiagnosticManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_diagnostic_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_diagnostic_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnosticProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_diagnostic_provider_load (provider));
    }

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static DexFuture *
foundry_diagnostic_manager_stop (FoundryService *service)
{
  FoundryDiagnosticManager *self = (FoundryDiagnosticManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_diagnostic_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_diagnostic_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnosticProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_diagnostic_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_diagnostic_manager_constructed (GObject *object)
{
  FoundryDiagnosticManager *self = (FoundryDiagnosticManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_diagnostic_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER,
                                         "context", context,
                                         NULL);
}

static void
foundry_diagnostic_manager_finalize (GObject *object)
{
  FoundryDiagnosticManager *self = (FoundryDiagnosticManager *)object;

  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_diagnostic_manager_parent_class)->finalize (object);
}

static void
foundry_diagnostic_manager_class_init (FoundryDiagnosticManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_diagnostic_manager_constructed;
  object_class->finalize = foundry_diagnostic_manager_finalize;

  service_class->start = foundry_diagnostic_manager_start;
  service_class->stop = foundry_diagnostic_manager_stop;
}

static void
foundry_diagnostic_manager_init (FoundryDiagnosticManager *self)
{
}

static DexFuture *
add_model_to_store (DexFuture *completed,
                    gpointer   user_data)
{
  g_autoptr(GListModel) model = NULL;
  GListStore *store = user_data;

  if ((model = dex_await_object (dex_ref (completed), NULL)))
    g_list_store_append (store, model);

  return dex_future_new_true ();
}

static gboolean
plugin_supports_language (PeasPluginInfo *plugin_info,
                          const char     *key,
                          const char     *language)
{
  const char *value;
  g_autofree char *delimit = NULL;
  g_auto(GStrv) parts = NULL;

  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (key != NULL);

  if (language == NULL)
    return TRUE;

  if (!(value = peas_plugin_info_get_external_data (plugin_info, key)) || !value[0])
    return TRUE;

  delimit = g_strdelimit (g_strdup (value), ",", ';');
  parts = g_strsplit (delimit, ";", 0);

  for (guint i = 0; parts[i]; i++)
    {
      if (g_str_equal (parts[i], language))
        return TRUE;
    }

  return FALSE;
}

static DexFuture *
foundry_diagnostic_manager_diagnose_fiber (FoundryDiagnosticManager *self,
                                           GFile                    *file,
                                           GBytes                   *contents,
                                           const char               *language)
{
  g_autoptr(FoundryInhibitor) inhibitor = NULL;
  g_autoptr(GListModel) flatten = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;
  GListModel *providers;
  DexFuture *all;
  guint n_items = 0;

  g_assert (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));
  g_assert (G_IS_FILE (file));

  /* Inhibit during diagnose */
  if (!(inhibitor = foundry_contextual_inhibit (FOUNDRY_CONTEXTUAL (self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* First make sure we're ready to run */
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (self)), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  futures = g_ptr_array_new_with_free_func (dex_unref);
  providers = G_LIST_MODEL (self->addins);
  n_items = g_list_model_get_n_items (providers);
  store = g_list_store_new (G_TYPE_LIST_MODEL);

  /* Asynchronously request diagnose from each provider and then post-process it
   * into a GListStore, which will be flattened later into each diagnostic set.
   *
   * We will give the user back immediately a list model they can await completion
   * on (which wraps the other models).
   */

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnosticProvider) provider = g_list_model_get_item (providers, i);
      g_autoptr(PeasPluginInfo) plugin_info = foundry_diagnostic_provider_dup_plugin_info (provider);
      DexFuture *future;

      if (language == NULL ||
          !plugin_supports_language (plugin_info, "Diagnostic-Provider-Languages", language))
        {
          FOUNDRY_TRACE_MSG ("Diagnose skipping `%s` due to language mismatch",
                             peas_plugin_info_get_module_name (plugin_info));
          continue;
        }

      future = foundry_diagnostic_provider_diagnose (provider, file, contents, language);
      future = dex_future_finally (future,
                                   add_model_to_store,
                                   g_object_ref (store),
                                   g_object_unref);

      g_ptr_array_add (futures, g_steal_pointer (&future));
    }

  if (futures->len > 0)
    all = dex_future_allv ((DexFuture **)futures->pdata, futures->len);
  else
    all = dex_future_new_true ();

  flatten = foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store)));
  foundry_list_model_set_future (flatten, all);

  return dex_future_new_take_object (g_steal_pointer (&flatten));
}


/**
 * foundry_diagnostic_manager_diagnose:
 * @self: a #FoundryDiagnosticManager
 * @file: (nullable): a #GFile
 * @contents: (nullable): optional #GBytes for file contents
 * @language: (nullable): the language identifier for @file
 *
 * Diagnoses @file using the loaded diagnostic providers and produces a
 * #DexFuture that will resolve to a [iface@Gio.ListModel] of
 * [class@Foundry.Diagnostic].
 *
 * The resulting [iface@Gio.ListModel] may be awaited for population to
 * complete using [func@Foundry.list_model_await].
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.Diagnostic].
 */
DexFuture *
foundry_diagnostic_manager_diagnose (FoundryDiagnosticManager *self,
                                     GFile                    *file,
                                     GBytes                   *contents,
                                     const char               *language)
{
  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_diagnostic_manager_diagnose_fiber),
                                  4,
                                  FOUNDRY_TYPE_DIAGNOSTIC_MANAGER, self,
                                  G_TYPE_FILE, file,
                                  G_TYPE_BYTES, contents,
                                  G_TYPE_STRING, language);
}

static DexFuture *
foundry_diagnostic_manager_diagnose_file_fiber (FoundryDiagnosticManager *self,
                                                FoundryInhibitor         *inhibitor,
                                                GFile                    *file,
                                                FoundryFileManager       *file_manager)
{
  g_autoptr(GBytes) contents = NULL;
  g_autofree char *language = NULL;

  g_assert (FOUNDRY_IS_FILE_MANAGER (file_manager));
  g_assert (FOUNDRY_IS_INHIBITOR (inhibitor));
  g_assert (G_IS_FILE (file));

  contents = dex_await_boxed (dex_file_load_contents_bytes (file), NULL);
  language = dex_await_string (foundry_file_manager_guess_language (file_manager, file, NULL, contents), NULL);

  return foundry_diagnostic_manager_diagnose (self, file, contents, language);
}

/**
 * foundry_diagnostic_manager_diagnose_file:
 * @self: a [class@Foundry.DiagnosticManager]
 * @file: a [iface@Gio.File]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] which may be awaited on for final
 *   completion of all diagnostics using [func@Foundry.list_model_await].
 */
DexFuture *
foundry_diagnostic_manager_diagnose_file (FoundryDiagnosticManager *self,
                                          GFile                    *file)
{
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundryInhibitor) inhibitor = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  if (!(inhibitor = foundry_contextual_inhibit (FOUNDRY_CONTEXTUAL (self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  context = foundry_inhibitor_dup_context (inhibitor);
  file_manager = foundry_context_dup_file_manager (context);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_diagnostic_manager_diagnose_file_fiber),
                                  4,
                                  FOUNDRY_TYPE_DIAGNOSTIC_MANAGER, self,
                                  FOUNDRY_TYPE_INHIBITOR, inhibitor,
                                  G_TYPE_FILE, file,
                                  FOUNDRY_TYPE_FILE_MANAGER, file_manager);
}

static DexFuture *
foundry_diagnostic_manager_diagnose_files_cb (DexFuture *completed,
                                              gpointer   user_data)
{
  FoundryDiagnosticManager *self = user_data;
  g_autoptr(GListModel) flatten = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(DexFuture) all = NULL;
  guint n_futures;

  g_assert (DEX_IS_FUTURE_SET (completed));
  g_assert (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_futures = dex_future_set_get_size (DEX_FUTURE_SET (completed));
  store = g_list_store_new (G_TYPE_LIST_MODEL);

  g_assert (n_futures > 0);

  for (guint i = 0; i < n_futures; i++)
    {
      g_autoptr(GError) error = NULL;
      DexFuture *future = dex_future_set_get_future_at (DEX_FUTURE_SET (completed), i);
      g_autoptr(GListModel) model = dex_await_object (dex_ref (future), &error);

      if (model == NULL)
        continue;

      g_assert (G_IS_LIST_MODEL (model));

      g_list_store_append (store, model);

      g_ptr_array_add (futures, foundry_list_model_await (model));
    }

  all = dex_future_allv ((DexFuture **)futures->pdata, futures->len);
  flatten = foundry_flatten_list_model_new (G_LIST_MODEL (g_steal_pointer (&store)));

  foundry_list_model_set_future (flatten, all);

  return dex_future_new_take_object (g_steal_pointer (&flatten));
}

/**
 * foundry_diagnostic_manager_diagnose_files:
 * @self: a [class@Foundry.DiagnosticManager]
 * @files: (array length=n_files): an array of [iface@Gio.File]
 * @n_files: number of @files
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] or %NULL
 */
DexFuture *
foundry_diagnostic_manager_diagnose_files (FoundryDiagnosticManager  *self,
                                           GFile                    **files,
                                           guint                      n_files)
{
  g_autoptr(GPtrArray) futures = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));
  dex_return_error_if_fail (files != NULL);
  dex_return_error_if_fail (n_files > 0);

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_files; i++)
    g_ptr_array_add (futures, foundry_diagnostic_manager_diagnose_file (self, files[i]));

  return dex_future_then (dex_future_anyv ((DexFuture **)futures->pdata, futures->len),
                          foundry_diagnostic_manager_diagnose_files_cb,
                          g_object_ref (self),
                          g_object_unref);
}

/**
 * foundry_diagnostic_manager_list_all:
 * @self: a [class@Foundry.DiagnosticManager]
 *
 * Lists all known diagnostics from all providers.
 *
 * This will call [method@Foundry.DiagnosticProvider.list_all] for every
 * available diagnostic provider.
 *
 * The list may update after the future resolves if providers implement
 * live updating of models.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.Diagnostic].
 */
DexFuture *
foundry_diagnostic_manager_list_all (FoundryDiagnosticManager *self)
{
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_MANAGER (self));

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnosticProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_diagnostic_provider_list_all (provider));
    }

  return _foundry_flatten_list_model_new_from_futures (futures);
}
