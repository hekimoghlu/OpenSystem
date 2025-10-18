/* plugin-lsp-bridge-diagnostic-provider.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
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

#include "foundry-lsp-client-private.h"

#include "plugin-lsp-bridge-diagnostic-provider.h"

struct _PluginLspBridgeDiagnosticProvider
{
  FoundryDiagnosticProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginLspBridgeDiagnosticProvider, plugin_lsp_bridge_diagnostic_provider, FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER)

typedef struct _Diagnose
{
  FoundryDiagnosticProvider *provider;
  GFile *file;
  GBytes *contents;
  char *language;
} Diagnose;

static void
diagnose_free (Diagnose *state)
{
  g_clear_object (&state->provider);
  g_clear_object (&state->file);
  g_clear_pointer (&state->contents, g_bytes_unref);
  g_clear_pointer (&state->language, g_free);
  g_free (state);
}

static DexFuture *
plugin_lsp_bridge_diagnostic_provider_diagnose_fiber (gpointer data)
{
  Diagnose *state = data;
  g_autoptr(FoundryLspManager) lsp_manager = NULL;
  g_autoptr(FoundryLspClient) client = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(JsonNode) params = NULL;
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *uri = NULL;
  GListModel *model;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_LSP_BRIDGE_DIAGNOSTIC_PROVIDER (state->provider));
  g_assert (G_IS_FILE (state->file));
  g_assert (state->language != NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (state->provider));
  lsp_manager = foundry_context_dup_lsp_manager (context);

  if (!(client = dex_await_object (foundry_lsp_manager_load_client (lsp_manager, state->language), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* TOOD: Synchronize buffer changes */

  uri = g_file_get_uri (state->file);

  params = FOUNDRY_JSON_OBJECT_NEW (
    "textDocument", "{",
      "uri", FOUNDRY_JSON_NODE_PUT_STRING (uri),
    "}"
  );

  /* In LSP 3.17 an option was added to query diagnostics specifically instead of
   * waiting for the peer to publish them periodically. This fits much better into
   * our design of diagnostics though may not be supported by all LSP servers.
   */

  store = g_list_store_new (G_TYPE_LIST_MODEL);

  if ((reply = dex_await_boxed (foundry_lsp_client_call (client, "textDocument/diagnostic", params), &error)))
    {
      /* TODO: Make list from @reply */
    }

  if ((model = _foundry_lsp_client_get_diagnostics (client, state->file)))
    g_list_store_append (store, model);

  return dex_future_new_take_object (foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store))));
}

static DexFuture *
plugin_lsp_bridge_diagnostic_provider_diagnose (FoundryDiagnosticProvider *provider,
                                                GFile                     *file,
                                                GBytes                    *contents,
                                                const char                *language)
{
  Diagnose *state;

  g_assert (PLUGIN_IS_LSP_BRIDGE_DIAGNOSTIC_PROVIDER (provider));
  g_assert (G_IS_FILE (file) || contents != NULL);

  if (language == NULL || file == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "Not supported");

  state = g_new0 (Diagnose, 1);
  state->provider = g_object_ref (provider);
  state->file = file ? g_object_ref (file) : NULL;
  state->contents = contents ? g_bytes_ref (contents) : NULL;
  state->language = g_strdup (language);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_lsp_bridge_diagnostic_provider_diagnose_fiber,
                              state,
                              (GDestroyNotify) diagnose_free);
}

static void
plugin_lsp_bridge_diagnostic_provider_class_init (PluginLspBridgeDiagnosticProviderClass *klass)
{
  FoundryDiagnosticProviderClass *diagnostic_provider_class = FOUNDRY_DIAGNOSTIC_PROVIDER_CLASS (klass);

  diagnostic_provider_class->diagnose = plugin_lsp_bridge_diagnostic_provider_diagnose;
}

static void
plugin_lsp_bridge_diagnostic_provider_init (PluginLspBridgeDiagnosticProvider *self)
{
}
