/* plugin-ollama-llm-completion.c
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

#include <gio/gio.h>

#include "plugin-ollama-llm-completion.h"
#include "plugin-ollama-llm-completion-chunk.h"

struct _PluginOllamaLlmCompletion
{
  FoundryLlmCompletion    parent_instance;
  FoundryJsonInputStream *stream;
  DexPromise             *finished;
};

G_DEFINE_FINAL_TYPE (PluginOllamaLlmCompletion, plugin_ollama_llm_completion, FOUNDRY_TYPE_LLM_COMPLETION)

static DexFuture *
plugin_ollama_llm_completion_panic (PluginOllamaLlmCompletion *self,
                                    GError                    *error)
{
  g_assert (PLUGIN_IS_OLLAMA_LLM_COMPLETION (self));
  g_assert (error != NULL);

  if (dex_future_is_pending (DEX_FUTURE (self->finished)))
    dex_promise_reject (self->finished, g_error_copy (error));

  return dex_future_new_for_error (g_steal_pointer (&error));
}

static DexFuture *
plugin_ollama_llm_completion_skip_cb (DexFuture *completed,
                                      gpointer   user_data)
{
  PluginOllamaLlmCompletionChunk *chunk = user_data;

  g_assert (PLUGIN_IS_OLLAMA_LLM_COMPLETION_CHUNK (chunk));

  return dex_future_new_take_object (g_object_ref (chunk));
}

static DexFuture *
plugin_ollama_llm_completion_next_chunk_cb (DexFuture *completed,
                                            gpointer   user_data)
{
  PluginOllamaLlmCompletion *self = user_data;
  g_autoptr(JsonNode) node = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (PLUGIN_IS_OLLAMA_LLM_COMPLETION (self));
  g_assert (G_IS_INPUT_STREAM (self->stream));

  if (!(node = dex_await_boxed (dex_ref (completed), &error)))
    return plugin_ollama_llm_completion_panic (self, g_steal_pointer (&error));

  return dex_future_finally (dex_input_stream_skip (G_INPUT_STREAM (self->stream), 1, G_PRIORITY_DEFAULT),
                             plugin_ollama_llm_completion_skip_cb,
                             plugin_ollama_llm_completion_chunk_new (node),
                             g_object_unref);
}

static DexFuture *
plugin_ollama_llm_completion_next_chunk (FoundryLlmCompletion *completion)
{
  PluginOllamaLlmCompletion *self = (PluginOllamaLlmCompletion *)completion;

  g_assert (PLUGIN_IS_OLLAMA_LLM_COMPLETION (self));
  g_assert (FOUNDRY_IS_JSON_INPUT_STREAM (self->stream));

  return dex_future_finally (foundry_json_input_stream_read_upto (self->stream, "\n", 1),
                             plugin_ollama_llm_completion_next_chunk_cb,
                             g_object_ref (self),
                             g_object_unref);
}

static DexFuture *
plugin_ollama_llm_completion_when_finished (FoundryLlmCompletion *completion)
{
  return dex_ref (PLUGIN_OLLAMA_LLM_COMPLETION (completion)->finished);
}

static void
plugin_ollama_llm_completion_finalize (GObject *object)
{
  PluginOllamaLlmCompletion *self = (PluginOllamaLlmCompletion *)object;

  g_clear_object (&self->stream);

  if (dex_future_is_pending (DEX_FUTURE (self->finished)))
    dex_promise_reject (self->finished,
                        g_error_new (G_IO_ERROR,
                                     G_IO_ERROR_CANCELLED,
                                     "Object disposed"));

  dex_clear (&self->finished);

  G_OBJECT_CLASS (plugin_ollama_llm_completion_parent_class)->finalize (object);
}

static void
plugin_ollama_llm_completion_class_init (PluginOllamaLlmCompletionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryLlmCompletionClass *llm_completion_class = FOUNDRY_LLM_COMPLETION_CLASS (klass);

  object_class->finalize = plugin_ollama_llm_completion_finalize;

  llm_completion_class->when_finished = plugin_ollama_llm_completion_when_finished;
  llm_completion_class->next_chunk = plugin_ollama_llm_completion_next_chunk;
}

static void
plugin_ollama_llm_completion_init (PluginOllamaLlmCompletion *self)
{
  self->finished = dex_promise_new ();
}

PluginOllamaLlmCompletion *
plugin_ollama_llm_completion_new (FoundryJsonInputStream *stream)
{
  PluginOllamaLlmCompletion *self;

  self = g_object_new (PLUGIN_TYPE_OLLAMA_LLM_COMPLETION, NULL);
  self->stream = g_object_ref (stream);

  return g_steal_pointer (&self);
}
