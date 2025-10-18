/* foundry-lsp-client.c
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

#include "foundry-diagnostic.h"
#include "foundry-json-node.h"
#include "foundry-jsonrpc-driver-private.h"
#include "foundry-lsp-client-private.h"
#include "foundry-lsp-provider.h"
#include "foundry-service-private.h"
#include "foundry-text-document.h"
#include "foundry-text-manager.h"
#include "foundry-util.h"

struct _FoundryLspClient
{
  FoundryContextual     parent_instance;
  FoundryLspProvider   *provider;
  FoundryJsonrpcDriver *driver;
  GSubprocess          *subprocess;
  DexFuture            *future;
  JsonNode             *capabilities;
  GHashTable           *diagnostics;
};

struct _FoundryLspClientClass
{
  FoundryContextualClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryLspClient, foundry_lsp_client, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_IO_STREAM,
  PROP_PROVIDER,
  PROP_SUBPROCESS,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static char *
translate_language_id (char *language_id)
{
  g_autofree char *freeme = language_id;

  if (g_str_equal (language_id, "python3"))
    return g_strdup ("python");

  return g_steal_pointer (&freeme);
}

static void
foundry_lsp_client_constructed (GObject *object)
{
  FoundryLspClient *self = (FoundryLspClient *)object;

  G_OBJECT_CLASS (foundry_lsp_client_parent_class)->constructed (object);

  if (self->driver)
    foundry_jsonrpc_driver_start (self->driver);
}

static void
foundry_lsp_client_finalize (GObject *object)
{
  FoundryLspClient *self = (FoundryLspClient *)object;

  if (self->driver != NULL)
    foundry_jsonrpc_driver_stop (self->driver);

  if (self->subprocess != NULL)
    g_subprocess_force_exit (self->subprocess);

  g_clear_object (&self->driver);
  g_clear_object (&self->provider);
  g_clear_object (&self->subprocess);
  g_clear_pointer (&self->capabilities, json_node_unref);
  g_clear_pointer (&self->diagnostics, g_hash_table_unref);
  dex_clear (&self->future);

  G_OBJECT_CLASS (foundry_lsp_client_parent_class)->finalize (object);
}

static void
foundry_lsp_client_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryLspClient *self = FOUNDRY_LSP_CLIENT (object);

  switch (prop_id)
    {
    case PROP_PROVIDER:
      g_value_set_object (value, self->provider);
      break;

    case PROP_SUBPROCESS:
      g_value_set_object (value, self->subprocess);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_lsp_client_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryLspClient *self = FOUNDRY_LSP_CLIENT (object);

  switch (prop_id)
    {
    case PROP_IO_STREAM:
      self->driver = foundry_jsonrpc_driver_new (g_value_get_object (value),
                                                 FOUNDRY_JSONRPC_STYLE_HTTP);
      break;

    case PROP_PROVIDER:
      self->provider = g_value_dup_object (value);
      break;

    case PROP_SUBPROCESS:
      if ((self->subprocess = g_value_dup_object (value)))
        self->future = dex_subprocess_wait_check (self->subprocess);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_lsp_client_class_init (FoundryLspClientClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_lsp_client_constructed;
  object_class->finalize = foundry_lsp_client_finalize;
  object_class->get_property = foundry_lsp_client_get_property;
  object_class->set_property = foundry_lsp_client_set_property;

  properties[PROP_IO_STREAM] =
    g_param_spec_object ("io-stream", NULL, NULL,
                         G_TYPE_IO_STREAM,
                         (G_PARAM_WRITABLE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_LSP_PROVIDER,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBPROCESS] =
    g_param_spec_object ("subprocess", NULL, NULL,
                         G_TYPE_SUBPROCESS,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_lsp_client_init (FoundryLspClient *self)
{
  self->diagnostics = g_hash_table_new_full ((GHashFunc) g_file_hash,
                                             (GEqualFunc) g_file_equal,
                                             g_object_unref,
                                             g_object_unref);
}

/**
 * foundry_lsp_client_query_capabilities:
 * @self: a #FoundryLspClient
 *
 * Queries the servers capabilities.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves when
 *   the query completes or fails.
 */
DexFuture *
foundry_lsp_client_query_capabilities (FoundryLspClient *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LSP_CLIENT (self));

  if (self->capabilities != NULL)
    return dex_future_new_take_boxed (JSON_TYPE_NODE, json_node_ref (self->capabilities));

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "not supported");
}

/**
 * foundry_lsp_client_call:
 * @self: a #FoundryLspClient
 * @method: the method name to call
 * @params: (nullable): parameters for the method call
 *
 * If @params is floating, the reference will be consumed.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves when
 *   a reply is received for the method call.
 */
DexFuture *
foundry_lsp_client_call (FoundryLspClient *self,
                         const char       *method,
                         JsonNode         *params)
{
  dex_return_error_if_fail (FOUNDRY_IS_LSP_CLIENT (self));
  dex_return_error_if_fail (method != NULL);

  return foundry_jsonrpc_driver_call (self->driver, method, params);
}

/**
 * foundry_lsp_client_notify:
 * @self: a #FoundryLspClient
 * @method: the method name to call
 * @params: (nullable): parameters for the notification
 *
 * If @params is floating, the reference will be consumed.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves when
 *   the notification has been sent.
 */
DexFuture *
foundry_lsp_client_notify (FoundryLspClient *self,
                           const char       *method,
                           JsonNode         *params)
{
  dex_return_error_if_fail (FOUNDRY_IS_LSP_CLIENT (self));
  dex_return_error_if_fail (method != NULL);

  return foundry_jsonrpc_driver_notify (self->driver, method, params);
}

static void
foundry_lsp_client_document_added (FoundryLspClient    *self,
                                   GFile               *file,
                                   FoundryTextDocument *document)
{
  g_autoptr(FoundryTextBuffer) buffer = NULL;
  g_autoptr(JsonNode) params = NULL;
  g_autoptr(GBytes) contents = NULL;
  g_autofree char *language_id = NULL;
  g_autofree char *uri = NULL;
  const char *text;
  gint64 change_count;

  g_assert (FOUNDRY_IS_LSP_CLIENT (self));
  g_assert (G_IS_FILE (file));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (document));

  buffer = foundry_text_document_dup_buffer (document);
  change_count = foundry_text_buffer_get_change_count (buffer);
  contents = foundry_text_buffer_dup_contents (buffer);
  language_id = foundry_text_buffer_dup_language_id (buffer);
  uri = foundry_text_document_dup_uri (document);

  if (foundry_str_empty0 (language_id))
    g_set_str (&language_id, "text/plain");

  language_id = translate_language_id (language_id);

  /* contents is \0 terminated */
  text = (const char *)g_bytes_get_data (contents, NULL);

  params = FOUNDRY_JSON_OBJECT_NEW (
    "textDocument", "{",
      "uri", FOUNDRY_JSON_NODE_PUT_STRING (uri),
      "languageId", FOUNDRY_JSON_NODE_PUT_STRING (language_id),
      "text", FOUNDRY_JSON_NODE_PUT_STRING (text),
      "version", FOUNDRY_JSON_NODE_PUT_INT (change_count),
    "}"
  );

  g_hash_table_replace (self->diagnostics,
                        g_file_dup (file),
                        g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC));

  dex_future_disown (foundry_lsp_client_notify (self, "textDocument/didOpen", params));
}

static void
foundry_lsp_client_document_removed (FoundryLspClient *self,
                                     GFile            *file)
{
  g_autoptr(JsonNode) params = NULL;
  g_autofree char *uri = NULL;

  g_assert (FOUNDRY_IS_LSP_CLIENT (self));
  g_assert (G_IS_FILE (file));

  uri = g_file_get_uri (file);

  g_hash_table_remove (self->diagnostics, file);

  params = FOUNDRY_JSON_OBJECT_NEW (
    "textDocument", "{",
      "uri", FOUNDRY_JSON_NODE_PUT_STRING (uri),
    "}"
  );

  dex_future_disown (foundry_lsp_client_notify (self, "textDocument/didClose", params));
}

static DexFuture *
foundry_lsp_client_load_fiber (gpointer data)
{
  FoundryLspClient *self = data;
  g_autoptr(FoundryTextManager) text_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(JsonNode) initialize_params = NULL;
  g_autoptr(JsonNode) initialization_options = NULL;
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autofree char *root_uri = NULL;
  g_autofree char *root_path = NULL;
  g_autofree char *basename = NULL;
  const char *trace_string = "off";
  JsonNode *caps = NULL;

  g_assert (FOUNDRY_IS_LSP_CLIENT (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  text_manager = foundry_context_dup_text_manager (context);
  project_dir = foundry_context_dup_project_directory (context);
  root_uri = g_file_get_uri (project_dir);
  basename = g_file_get_basename (project_dir);

  if (g_file_is_native (project_dir))
    root_path = g_file_get_path (project_dir);
  else
    root_path = g_strdup ("");

  if (self->provider != NULL)
    initialization_options = foundry_lsp_provider_dup_initialization_options (self->provider);

  initialize_params = FOUNDRY_JSON_OBJECT_NEW (
#if 0
    /* Some LSPs will monitor the PID of the editor and exit when they
     * detect the editor has exited. Since we are likely in a different
     * PID namespace than the LSP, there is a PID mismatch and it will
     * probably get PID 2 (from Flatpak) and not be of any use.
     *
     * Just ignore it as the easiest solution.
     *
     * If this causes problems elsewhere, we might need to try to setup
     * a quirk handler for some LSPs.
     *
     * https://gitlab.gnome.org/GNOME/gnome-builder/-/issues/2050
     */
    "processId", FOUNDRY_JSON_NODE_PUT_INT (getpid ()),
#endif
    "rootUri", FOUNDRY_JSON_NODE_PUT_STRING (root_uri),
    "clientInfo", "{",
      "name", FOUNDRY_JSON_NODE_PUT_STRING ("Foundry"),
      "version", FOUNDRY_JSON_NODE_PUT_STRING (PACKAGE_VERSION),
    "}",
    "rootPath", FOUNDRY_JSON_NODE_PUT_STRING (root_path),
    "workspaceFolders", "[",
      "{",
        "uri", FOUNDRY_JSON_NODE_PUT_STRING (root_uri),
        "name", FOUNDRY_JSON_NODE_PUT_STRING (basename),
      "}",
    "]",
    "trace", FOUNDRY_JSON_NODE_PUT_STRING (trace_string),
    "capabilities", "{",
      "workspace", "{",
        "applyEdit", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
        "configuration", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
        "symbol", "{",
          "SymbolKind", "{",
            "valueSet", "[",
              FOUNDRY_JSON_NODE_PUT_INT (1), /* File */
              FOUNDRY_JSON_NODE_PUT_INT (2), /* Module */
              FOUNDRY_JSON_NODE_PUT_INT (3), /* Namespace */
              FOUNDRY_JSON_NODE_PUT_INT (4), /* Package */
              FOUNDRY_JSON_NODE_PUT_INT (5), /* Class */
              FOUNDRY_JSON_NODE_PUT_INT (6), /* Method */
              FOUNDRY_JSON_NODE_PUT_INT (7), /* Property */
              FOUNDRY_JSON_NODE_PUT_INT (8), /* Field */
              FOUNDRY_JSON_NODE_PUT_INT (9), /* Constructor */
              FOUNDRY_JSON_NODE_PUT_INT (10), /* Enum */
              FOUNDRY_JSON_NODE_PUT_INT (11), /* Interface */
              FOUNDRY_JSON_NODE_PUT_INT (12), /* Function */
              FOUNDRY_JSON_NODE_PUT_INT (13), /* Variable */
              FOUNDRY_JSON_NODE_PUT_INT (14), /* Constant */
              FOUNDRY_JSON_NODE_PUT_INT (15), /* String */
              FOUNDRY_JSON_NODE_PUT_INT (16), /* Number */
              FOUNDRY_JSON_NODE_PUT_INT (17), /* Boolean */
              FOUNDRY_JSON_NODE_PUT_INT (18), /* Array */
              FOUNDRY_JSON_NODE_PUT_INT (19), /* Object */
              FOUNDRY_JSON_NODE_PUT_INT (20), /* Key */
              FOUNDRY_JSON_NODE_PUT_INT (21), /* Null */
              FOUNDRY_JSON_NODE_PUT_INT (22), /* EnumMember */
              FOUNDRY_JSON_NODE_PUT_INT (23), /* Struct */
              FOUNDRY_JSON_NODE_PUT_INT (24), /* Event */
              FOUNDRY_JSON_NODE_PUT_INT (25), /* Operator */
              FOUNDRY_JSON_NODE_PUT_INT (26), /* TypeParameter */
            "]",
          "}",
        "}",
      "}",
      "textDocument", "{",
        "completion", "{",
          "contextSupport", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
          "completionItem", "{",
            "snippetSupport", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
            "documentationFormat", "[",
              "markdown",
              "plaintext",
            "]",
            "deprecatedSupport", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
            "labelDetailsSupport", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
          "}",
          "completionItemKind", "{",
            "valueSet", "[",
              FOUNDRY_JSON_NODE_PUT_INT (1),
              FOUNDRY_JSON_NODE_PUT_INT (2),
              FOUNDRY_JSON_NODE_PUT_INT (3),
              FOUNDRY_JSON_NODE_PUT_INT (4),
              FOUNDRY_JSON_NODE_PUT_INT (5),
              FOUNDRY_JSON_NODE_PUT_INT (6),
              FOUNDRY_JSON_NODE_PUT_INT (7),
              FOUNDRY_JSON_NODE_PUT_INT (8),
              FOUNDRY_JSON_NODE_PUT_INT (9),
              FOUNDRY_JSON_NODE_PUT_INT (10),
              FOUNDRY_JSON_NODE_PUT_INT (11),
              FOUNDRY_JSON_NODE_PUT_INT (12),
              FOUNDRY_JSON_NODE_PUT_INT (13),
              FOUNDRY_JSON_NODE_PUT_INT (14),
              FOUNDRY_JSON_NODE_PUT_INT (15),
              FOUNDRY_JSON_NODE_PUT_INT (16),
              FOUNDRY_JSON_NODE_PUT_INT (17),
              FOUNDRY_JSON_NODE_PUT_INT (18),
              FOUNDRY_JSON_NODE_PUT_INT (19),
              FOUNDRY_JSON_NODE_PUT_INT (20),
              FOUNDRY_JSON_NODE_PUT_INT (21),
              FOUNDRY_JSON_NODE_PUT_INT (22),
              FOUNDRY_JSON_NODE_PUT_INT (23),
              FOUNDRY_JSON_NODE_PUT_INT (24),
              FOUNDRY_JSON_NODE_PUT_INT (25),
            "]",
          "}",
        "}",
        "diagnostic", "{",
        "}",
        "hover", "{",
          "contentFormat", "[",
            "markdown",
            "plaintext",
          "]",
        "}",
        "publishDiagnostics", "{",
          "tagSupport", "{",
            "valueSet", "[",
              FOUNDRY_JSON_NODE_PUT_INT (1),
              FOUNDRY_JSON_NODE_PUT_INT (2),
            "]",
          "}",
        "}",
        "codeAction", "{",
          "dynamicRegistration", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
          "isPreferredSupport", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
          "codeActionLiteralSupport", "{",
            "codeActionKind", "{",
              "valueSet", "[",
                "",
                "quickfix",
                "refactor",
                "refactor.extract",
                "refactor.inline",
                "refactor.rewrite",
                "source",
                "source.organizeImports",
              "]",
            "}",
          "}",
        "}",
      "}",
      "window", "{",
        "workDoneProgress", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "}",
    "}",
    "initializationOptions", FOUNDRY_JSON_NODE_PUT_NODE (initialization_options)
  );

  if (!(reply = dex_await_boxed (foundry_jsonrpc_driver_call (self->driver, "initialize", initialize_params), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (FOUNDRY_JSON_OBJECT_PARSE (reply, "capabilities", FOUNDRY_JSON_NODE_GET_NODE (&caps)))
    self->capabilities = json_node_ref (caps);

  g_signal_connect_object (text_manager,
                           "document-added",
                           G_CALLBACK (foundry_lsp_client_document_added),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (text_manager,
                           "document-removed",
                           G_CALLBACK (foundry_lsp_client_document_removed),
                           self,
                           G_CONNECT_SWAPPED);

  /* Notify LSP of open documents */
  if (dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (text_manager)), NULL))
    {
      g_autoptr(GListModel) documents = foundry_text_manager_list_documents (text_manager);
      guint n_items = g_list_model_get_n_items (documents);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryTextDocument) document = g_list_model_get_item (documents, i);
          g_autoptr(GFile) file = foundry_text_document_dup_file (document);

          foundry_lsp_client_document_added (self, file, document);
        }
    }

  return dex_future_new_true ();
}

DexFuture *
foundry_lsp_client_new (FoundryContext *context,
                        GIOStream      *io_stream,
                        GSubprocess    *subprocess)
{
  g_autoptr(FoundryLspClient) client = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));
  dex_return_error_if_fail (G_IS_IO_STREAM (io_stream));
  dex_return_error_if_fail (!subprocess || G_IS_SUBPROCESS (subprocess));

  client = g_object_new (FOUNDRY_TYPE_LSP_CLIENT,
                         "context", context,
                         "io-stream", io_stream,
                         "subprocess", subprocess,
                         NULL);

  return dex_future_then (dex_scheduler_spawn (NULL, 0,
                                               foundry_lsp_client_load_fiber,
                                               g_object_ref (client),
                                               g_object_unref),
                          foundry_future_return_object,
                          g_object_ref (client),
                          g_object_unref);
}

/**
 * foundry_lsp_client_await:
 * @self: a [class@Foundry.LspClient]
 *
 * Await completion of the client subprocess.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   when the subprocess has exited or rejects with error.
 */
DexFuture *
foundry_lsp_client_await (FoundryLspClient *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LSP_CLIENT (self));

  if (self->future == NULL)
    return dex_future_new_true ();

  return dex_ref (self->future);
}

gboolean
foundry_lsp_client_supports_language (FoundryLspClient *self,
                                      const char       *language_id)
{
  g_autoptr(PeasPluginInfo) plugin_info = NULL;

  g_return_val_if_fail (FOUNDRY_IS_LSP_CLIENT (self), FALSE);
  g_return_val_if_fail (language_id != NULL, FALSE);

  if ((plugin_info = foundry_lsp_provider_dup_plugin_info (self->provider)))
    {
      const char *x_languages = peas_plugin_info_get_external_data (plugin_info, "LSP-Languages");
      g_auto(GStrv) languages = g_strsplit (x_languages, ";", 0);

      for (guint i = 0; languages[i]; i++)
        {
          if (g_strcmp0 (languages[i], language_id) == 0)
            return TRUE;
        }
    }

  return FALSE;
}

GListModel *
_foundry_lsp_client_get_diagnostics (FoundryLspClient *self,
                                     GFile            *file)
{
  g_return_val_if_fail (FOUNDRY_IS_LSP_CLIENT (self), NULL);
  g_return_val_if_fail (G_IS_FILE (file), NULL);

  return g_hash_table_lookup (self->diagnostics, file);
}
