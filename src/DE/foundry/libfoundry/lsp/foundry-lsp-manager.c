/* foundry-lsp-manager.c
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
#include "foundry-build-pipeline.h"
#include "foundry-context.h"
#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-model-manager.h"
#include "foundry-lsp-client.h"
#include "foundry-lsp-manager.h"
#include "foundry-lsp-provider-private.h"
#include "foundry-lsp-server.h"
#include "foundry-process-launcher.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

struct _FoundryLspManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
  GListModel       *flatten;
  GHashTable       *clients_by_module_name;
};

struct _FoundryLspManagerClass
{
  FoundryServiceClass parent_class;
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryLspManager, foundry_lsp_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static FoundryLspProvider *
foundry_lsp_manager_dup_preferred_provider (FoundryLspManager *self,
                                            const char        *language_id)
{
  g_autoptr(FoundryLspProvider) best = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autofree char *preferred = NULL;

  g_return_val_if_fail (FOUNDRY_IS_LSP_MANAGER (self), NULL);
  g_return_val_if_fail (language_id != NULL, NULL);

  settings = foundry_lsp_manager_load_language_settings (self, language_id);
  preferred = foundry_settings_get_string (settings, "preferred-module-name");

  if (self->addins != NULL)
    {
      guint n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryLspProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
          g_autoptr(PeasPluginInfo) plugin_info = foundry_lsp_provider_dup_plugin_info (provider);
          g_autoptr(FoundryLspServer) server = foundry_lsp_provider_dup_server (provider);

          if (preferred != NULL &&
              server != NULL &&
              plugin_info != NULL &&
              g_strcmp0 (preferred, peas_plugin_info_get_module_name (plugin_info)) == 0)
            return g_steal_pointer (&provider);

          if (best == NULL &&
              server != NULL &&
              foundry_lsp_server_supports_language (server, language_id))
            best = g_steal_pointer (&provider);
        }
    }

  return g_steal_pointer (&best);
}

static void
foundry_lsp_manager_provider_added (PeasExtensionSet *set,
                                    PeasPluginInfo   *plugin_info,
                                    GObject          *addin,
                                    gpointer          user_data)
{
  FoundryLspManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_LSP_PROVIDER (addin));
  g_assert (FOUNDRY_IS_LSP_MANAGER (self));

  g_debug ("Adding FoundryLspProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_lsp_provider_load (FOUNDRY_LSP_PROVIDER (addin)));
}

static void
foundry_lsp_manager_provider_removed (PeasExtensionSet *set,
                                      PeasPluginInfo   *plugin_info,
                                      GObject          *addin,
                                      gpointer          user_data)
{
  FoundryLspManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_LSP_PROVIDER (addin));
  g_assert (FOUNDRY_IS_LSP_MANAGER (self));

  g_debug ("Removing FoundryLspProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_lsp_provider_unload (FOUNDRY_LSP_PROVIDER (addin)));
}

static DexFuture *
foundry_lsp_manager_start (FoundryService *service)
{
  FoundryLspManager *self = (FoundryLspManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_lsp_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_lsp_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLspProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_lsp_provider_load (provider));
    }

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static DexFuture *
foundry_lsp_manager_stop (FoundryService *service)
{
  FoundryLspManager *self = (FoundryLspManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_lsp_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_lsp_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLspProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_lsp_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_lsp_manager_constructed (GObject *object)
{
  FoundryLspManager *self = (FoundryLspManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_lsp_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_LSP_PROVIDER,
                                         "context", context,
                                         NULL);

  g_object_set (self->flatten,
                "model", self->addins,
                NULL);
}

static void
foundry_lsp_manager_finalize (GObject *object)
{
  FoundryLspManager *self = (FoundryLspManager *)object;

  g_clear_pointer (&self->clients_by_module_name, g_hash_table_unref);
  g_clear_object (&self->flatten);
  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_lsp_manager_parent_class)->finalize (object);
}

static void
foundry_lsp_manager_class_init (FoundryLspManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_lsp_manager_constructed;
  object_class->finalize = foundry_lsp_manager_finalize;

  service_class->start = foundry_lsp_manager_start;
  service_class->stop = foundry_lsp_manager_stop;
}

static void
foundry_lsp_manager_init (FoundryLspManager *self)
{
  self->clients_by_module_name = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, dex_unref);
  self->flatten = foundry_flatten_list_model_new (NULL);

  g_signal_connect_object (self->flatten,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

typedef struct _RunClient
{
  GWeakRef  self_wr;
  char     *module_name;
} RunClient;

static void
run_client_free (RunClient *state)
{
  g_weak_ref_clear (&state->self_wr);
  g_clear_pointer (&state->module_name, g_free);
  g_free (state);
}

static DexFuture *
foundry_lsp_manager_reap_client (DexFuture *completed,
                                 gpointer   user_data)
{
  RunClient *state = user_data;
  g_autoptr(FoundryLspManager) self = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (state != NULL);
  g_assert (state->module_name != NULL);

  if ((self = g_weak_ref_get (&state->self_wr)))
    {
      if (self->clients_by_module_name != NULL)
        g_hash_table_remove (self->clients_by_module_name, state->module_name);
    }

  return dex_future_new_true ();
}

typedef struct _LoadClient
{
  FoundryLspManager *self;
  FoundryLspServer  *server;
  char              *module_name;
  int                stdin_fd;
  int                stdout_fd;
} LoadClient;

static void
load_client_free (LoadClient *state)
{
  DexFuture *future;

  /* Remove client if it has failed to initialize */
  if (state->self->clients_by_module_name != NULL &&
      (future = g_hash_table_lookup (state->self->clients_by_module_name, state->module_name)) &&
      dex_future_is_rejected (future))
    g_hash_table_remove (state->self->clients_by_module_name, state->module_name);

  g_clear_object (&state->self);
  g_clear_object (&state->server);
  g_clear_pointer (&state->module_name, g_free);
  g_clear_fd (&state->stdin_fd, NULL);
  g_clear_fd (&state->stdout_fd, NULL);

  g_free (state);
}

static DexFuture *
foundry_lsp_manager_load_client_fiber (gpointer data)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryLspClient) client = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GIOStream) io_stream = NULL;
  g_autoptr(GError) error = NULL;
  GSubprocessFlags flags = 0;
  LoadClient *state = data;
  RunClient *run;
  gboolean log_stderr;

  g_assert (FOUNDRY_IS_LSP_MANAGER (state->self));
  g_assert (FOUNDRY_IS_LSP_SERVER (state->server));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (state->self))))
    return foundry_future_new_disposed ();

  build_manager = foundry_context_dup_build_manager (context);
  pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), NULL);

  settings = foundry_context_load_settings (context, "app.devsuite.foundry.lsp", NULL);
  log_stderr = foundry_settings_get_boolean (settings, "log-stderr");

  if (!log_stderr)
    flags |= G_SUBPROCESS_FLAGS_STDERR_SILENCE;

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_lsp_server_prepare (state->server, pipeline, launcher), &error) ||
      !(io_stream = foundry_process_launcher_create_stdio_stream (launcher, &error)) ||
      !(subprocess = foundry_process_launcher_spawn_with_flags (launcher, flags, &error)) ||
      !(client = dex_await_object (foundry_lsp_client_new (context, io_stream, subprocess), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  run = g_new0 (RunClient, 1);
  run->module_name = g_strdup (state->module_name);
  g_weak_ref_init (&run->self_wr, state->self);

  dex_future_disown (dex_future_finally (foundry_lsp_client_await (client),
                                         foundry_lsp_manager_reap_client,
                                         run,
                                         (GDestroyNotify) run_client_free));

  return dex_future_new_take_object (g_steal_pointer (&client));
}

/**
 * foundry_lsp_manager_load_client:
 * @self: a #FoundryLspManager
 *
 * Loads a [class@Foundry.LspClient] for the @language_id.
 *
 * If an existing client is already created for this language,
 * that client will be returned.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.LspClient].
 */
DexFuture *
foundry_lsp_manager_load_client (FoundryLspManager *self,
                                 const char        *language_id)
{
  g_autoptr(FoundryLspProvider) provider = NULL;
  g_autoptr(FoundryLspServer) server = NULL;
  g_autoptr(PeasPluginInfo) plugin_info = NULL;
  const char *module_name;
  LoadClient *state;
  DexFuture *future;

  dex_return_error_if_fail (FOUNDRY_IS_LSP_MANAGER (self));
  dex_return_error_if_fail (language_id != NULL);

  if (!(provider = foundry_lsp_manager_dup_preferred_provider (self, language_id)))
    goto lookup_failure;

  if (!(plugin_info = foundry_lsp_provider_dup_plugin_info (provider)))
    goto lookup_failure;

  if (!(server = foundry_lsp_provider_dup_server (provider)))
    goto lookup_failure;

  module_name = peas_plugin_info_get_module_name (plugin_info);

  if ((future = g_hash_table_lookup (self->clients_by_module_name, module_name)))
    return dex_ref (future);

  state = g_new0 (LoadClient, 1);
  state->self = g_object_ref (self);
  state->server = g_object_ref (server);
  state->module_name = g_strdup (module_name);
  state->stdin_fd = -1;
  state->stdout_fd = -1;

  future = dex_scheduler_spawn (NULL, 0,
                                foundry_lsp_manager_load_client_fiber,
                                state,
                                (GDestroyNotify) load_client_free);

  g_hash_table_replace (self->clients_by_module_name,
                        g_strdup (module_name),
                        dex_ref (future));

  return future;

lookup_failure:
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "The language \"%s\" does not have a supported LSP",
                                language_id);
}

static GType
foundry_lsp_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_LSP_SERVER;
}

static guint
foundry_lsp_manager_get_n_items (GListModel *model)
{
  FoundryLspManager *self = FOUNDRY_LSP_MANAGER (model);

  return g_list_model_get_n_items (G_LIST_MODEL (self->flatten));
}

static gpointer
foundry_lsp_manager_get_item (GListModel *model,
                              guint       position)
{
  FoundryLspManager *self = FOUNDRY_LSP_MANAGER (model);

  return g_list_model_get_item (G_LIST_MODEL (self->flatten), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_lsp_manager_get_item_type;
  iface->get_n_items = foundry_lsp_manager_get_n_items;
  iface->get_item = foundry_lsp_manager_get_item;
}

/**
 * foundry_lsp_manager_load_language_settings:
 * @self: a [class@Foundry.LspManager]
 * @language_id: the language identifier
 *
 * Returns: (transfer full): a [class@Foundry.Settings]
 */
FoundrySettings *
foundry_lsp_manager_load_language_settings (FoundryLspManager *self,
                                            const char        *language_id)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *path = NULL;

  g_return_val_if_fail (FOUNDRY_IS_LSP_MANAGER (self), NULL);
  g_return_val_if_fail (language_id != NULL, NULL);
  g_return_val_if_fail (strchr (language_id, '/') == NULL, NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  path = g_strdup_printf ("/app/devsuite/foundry/lsp/language/%s/", language_id);

  return foundry_context_load_settings (context, "app.devsuite.foundry.lsp.language", path);
}
