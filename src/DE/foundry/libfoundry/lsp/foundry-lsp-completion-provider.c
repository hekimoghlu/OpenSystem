/* foundry-lsp-completion-provider.c
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

#include <json-glib/json-glib.h>

#include "foundry-completion-request.h"
#include "foundry-json-node.h"
#include "foundry-lsp-capabilities-private.h"
#include "foundry-lsp-client.h"
#include "foundry-lsp-completion-provider.h"
#include "foundry-lsp-completion-results-private.h"
#include "foundry-lsp-manager.h"
#include "foundry-text-iter.h"
#include "foundry-util.h"

struct _FoundryLspCompletionProvider
{
  FoundryCompletionProvider  parent_instance;
  FoundryLspClient          *client;
  char                      *trigger_chars;
};

G_DEFINE_FINAL_TYPE (FoundryLspCompletionProvider, foundry_lsp_completion_provider, FOUNDRY_TYPE_COMPLETION_PROVIDER)

static DexFuture *
foundry_lsp_completion_provider_load_client (FoundryLspCompletionProvider *self,
                                             const char                   *language_id)
{
  g_autoptr(FoundryLspManager) lsp_manager = NULL;
  g_autoptr(FoundryLspClient) client = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_LSP_COMPLETION_PROVIDER (self));
  dex_return_error_if_fail (language_id != NULL);

  if (self->client != NULL)
    return dex_future_new_take_object (g_object_ref (self->client));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))) &&
      (lsp_manager = foundry_context_dup_lsp_manager (context)) &&
      (client = dex_await_object (foundry_lsp_manager_load_client (lsp_manager, language_id), &error)))
    {
      g_autoptr(JsonNode) capabilities = NULL;
      g_auto(GStrv) trigger_chars = NULL;

      /* Keep a copy of the client for later to do trigger checks. */
      /* TODO: track failures to clear client pointer */
      g_set_object (&self->client, client);

      if ((capabilities = dex_await_boxed (foundry_lsp_client_query_capabilities (client), NULL)))
        {
          if (FOUNDRY_JSON_OBJECT_PARSE (capabilities,
                                         "completionProvider", "{",
                                           "triggerCharacters", FOUNDRY_JSON_NODE_GET_STRV (&trigger_chars),
                                         "}"))
            {
              g_autoptr(GString) str = g_string_new (NULL);
              for (guint i = 0; trigger_chars[i]; i++)
                g_string_append_unichar (str, g_utf8_get_char (trigger_chars[i]));
              g_set_str (&self->trigger_chars, str->str);
            }
        }

      return dex_future_new_take_object (g_object_ref (client));
    }

  if (error != NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_future_new_disposed ();
}

static DexFuture *
foundry_lsp_completion_provider_complete_fiber (FoundryLspCompletionProvider *self,
                                                FoundryCompletionRequest     *request)
{
  g_autoptr(FoundryLspClient) client = NULL;
  g_autofree char *language_id = NULL;

  g_assert (FOUNDRY_IS_LSP_COMPLETION_PROVIDER (self));
  g_assert (FOUNDRY_IS_COMPLETION_REQUEST (request));

  if ((language_id = foundry_completion_request_dup_language_id (request)))
    {
      g_autoptr(JsonNode) capabilities = NULL;
      g_autoptr(JsonNode) params = NULL;
      g_autoptr(JsonNode) reply = NULL;
      g_autoptr(GError) error = NULL;
      g_autoptr(GFile) file = NULL;
      g_autofree char *uri = NULL;
      g_autofree char *typed_text = NULL;
      FoundryCompletionActivation activation;
      FoundryTextIter begin;
      FoundryTextIter end;
      int trigger_kind;
      int line;
      int line_offset;

      if (!(file = foundry_completion_request_dup_file (request)))
        return foundry_future_new_disposed ();

      if (!(client = dex_await_object (foundry_lsp_completion_provider_load_client (self, language_id), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!(capabilities = dex_await_boxed (foundry_lsp_client_query_capabilities (client), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!foundry_lsp_capabilities_can_complete (capabilities))
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_NOT_SUPPORTED,
                                      "Not supported");

      g_assert (language_id != NULL);
      g_assert (FOUNDRY_IS_LSP_CLIENT (client));

      typed_text = foundry_completion_request_dup_word (request);

      foundry_completion_request_get_bounds (request, &begin, &end);

      activation = foundry_completion_request_get_activation (request);

      if (activation == FOUNDRY_COMPLETION_ACTIVATION_INTERACTIVE)
        trigger_kind = 2;
      else
        trigger_kind = 1;

      line = foundry_text_iter_get_line (&begin);
      line_offset = foundry_text_iter_get_line_offset (&begin);
      uri = g_file_get_uri (file);

      params = FOUNDRY_JSON_OBJECT_NEW (
        "textDocument", "{",
          "uri", FOUNDRY_JSON_NODE_PUT_STRING (uri),
        "}",
        "position", "{",
          "line", FOUNDRY_JSON_NODE_PUT_INT (line),
          "character", FOUNDRY_JSON_NODE_PUT_INT (line_offset),
        "}",
        "context", "{",
          "triggerKind", FOUNDRY_JSON_NODE_PUT_INT (trigger_kind),
        "}"
      );

      if (!(reply = dex_await_boxed (foundry_lsp_client_call (client, "textDocument/completion", params), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      return foundry_lsp_completion_results_new (client, reply, typed_text);
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

static DexFuture *
foundry_lsp_completion_provider_complete (FoundryCompletionProvider *provider,
                                          FoundryCompletionRequest  *request)
{
  g_assert (FOUNDRY_IS_LSP_COMPLETION_PROVIDER (provider));
  g_assert (FOUNDRY_IS_COMPLETION_REQUEST (request));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_lsp_completion_provider_complete_fiber),
                                  2,
                                  FOUNDRY_TYPE_LSP_COMPLETION_PROVIDER, provider,
                                  FOUNDRY_TYPE_COMPLETION_REQUEST, request);
}

static gboolean
foundry_lsp_completion_provider_is_trigger (FoundryCompletionProvider *provider,
                                            const FoundryTextIter     *iter,
                                            gunichar                   ch)
{
  FoundryLspCompletionProvider *self = FOUNDRY_LSP_COMPLETION_PROVIDER (provider);

  if (self->trigger_chars != NULL)
    {
      for (const char *c = self->trigger_chars; *c; c = g_utf8_next_char (c))
        {
          if (ch == g_utf8_get_char (c))
            return TRUE;
        }
    }

  return FALSE;
}

static DexFuture *
foundry_lsp_completion_provider_refilter (FoundryCompletionProvider *provider,
                                          FoundryCompletionRequest  *request,
                                          GListModel                *model)
{
  g_autofree char *typed_text = NULL;

  g_assert (FOUNDRY_IS_LSP_COMPLETION_PROVIDER (provider));
  g_assert (FOUNDRY_IS_COMPLETION_REQUEST (request));
  g_assert (FOUNDRY_IS_LSP_COMPLETION_RESULTS (model));

  typed_text = foundry_completion_request_dup_word (request);

  foundry_lsp_completion_results_refilter (FOUNDRY_LSP_COMPLETION_RESULTS (model), typed_text);

  return dex_future_new_take_object (g_object_ref (model));
}

static void
foundry_lsp_completion_provider_dispose (GObject *object)
{
  FoundryLspCompletionProvider *self = (FoundryLspCompletionProvider *)object;

  g_clear_object (&self->client);
  g_clear_pointer (&self->trigger_chars, g_free);

  G_OBJECT_CLASS (foundry_lsp_completion_provider_parent_class)->dispose (object);
}

static void
foundry_lsp_completion_provider_class_init (FoundryLspCompletionProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCompletionProviderClass *completion_provider_class = FOUNDRY_COMPLETION_PROVIDER_CLASS (klass);

  object_class->dispose = foundry_lsp_completion_provider_dispose;

  completion_provider_class->complete = foundry_lsp_completion_provider_complete;
  completion_provider_class->refilter = foundry_lsp_completion_provider_refilter;
  completion_provider_class->is_trigger = foundry_lsp_completion_provider_is_trigger;
}

static void
foundry_lsp_completion_provider_init (FoundryLspCompletionProvider *self)
{
}
