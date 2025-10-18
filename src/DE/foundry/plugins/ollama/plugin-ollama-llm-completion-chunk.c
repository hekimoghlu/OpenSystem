/* plugin-ollama-llm-completion-chunk.c
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

#include "plugin-ollama-llm-completion-chunk.h"

struct _PluginOllamaLlmCompletionChunk
{
  FoundryLlmCompletionChunk parent_instance;
  JsonNode *node;
};

enum {
  PROP_0,
  PROP_NODE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginOllamaLlmCompletionChunk, plugin_ollama_llm_completion_chunk, FOUNDRY_TYPE_LLM_COMPLETION_CHUNK)

static GParamSpec *properties[N_PROPS];

static char *
plugin_ollama_llm_completion_chunk_dup_text (FoundryLlmCompletionChunk *chunk)
{
  PluginOllamaLlmCompletionChunk *self = (PluginOllamaLlmCompletionChunk *)chunk;

  g_assert (PLUGIN_IS_OLLAMA_LLM_COMPLETION_CHUNK (self));
  g_assert (self->node != NULL);

  if (JSON_NODE_HOLDS_OBJECT (self->node))
    {
      JsonObject *obj = json_node_get_object (self->node);
      JsonNode *response;

      if (json_object_has_member (obj, "response") &&
          (response = json_object_get_member (obj, "response")) &&
          json_node_get_value_type (response) == G_TYPE_STRING)
        return g_strdup (json_node_get_string (response));
    }

  return NULL;
}

static gboolean
plugin_ollama_llm_completion_chunk_is_done (FoundryLlmCompletionChunk *chunk)
{
  PluginOllamaLlmCompletionChunk *self = (PluginOllamaLlmCompletionChunk *)chunk;

  g_assert (PLUGIN_IS_OLLAMA_LLM_COMPLETION_CHUNK (self));
  g_assert (self->node != NULL);

  if (JSON_NODE_HOLDS_OBJECT (self->node))
    {
      JsonObject *obj = json_node_get_object (self->node);
      JsonNode *node;

      if (json_object_has_member (obj, "done") &&
          (node = json_object_get_member (obj, "done")) &&
          json_node_get_value_type (node) == G_TYPE_BOOLEAN)
        return json_node_get_boolean (node);
    }

  return TRUE;
}

static void
plugin_ollama_llm_completion_chunk_finalize (GObject *object)
{
  PluginOllamaLlmCompletionChunk *self = (PluginOllamaLlmCompletionChunk *)object;

  g_clear_pointer (&self->node, json_node_unref);

  G_OBJECT_CLASS (plugin_ollama_llm_completion_chunk_parent_class)->finalize (object);
}

static void
plugin_ollama_llm_completion_chunk_get_property (GObject    *object,
                                                 guint       prop_id,
                                                 GValue     *value,
                                                 GParamSpec *pspec)
{
  PluginOllamaLlmCompletionChunk *self = PLUGIN_OLLAMA_LLM_COMPLETION_CHUNK (object);

  switch (prop_id)
    {
    case PROP_NODE:
      g_value_set_boxed (value, self->node);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_ollama_llm_completion_chunk_set_property (GObject      *object,
                                                 guint         prop_id,
                                                 const GValue *value,
                                                 GParamSpec   *pspec)
{
  PluginOllamaLlmCompletionChunk *self = PLUGIN_OLLAMA_LLM_COMPLETION_CHUNK (object);

  switch (prop_id)
    {
    case PROP_NODE:
      self->node = g_value_dup_boxed (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_ollama_llm_completion_chunk_class_init (PluginOllamaLlmCompletionChunkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryLlmCompletionChunkClass *chunk_class = FOUNDRY_LLM_COMPLETION_CHUNK_CLASS (klass);

  object_class->finalize = plugin_ollama_llm_completion_chunk_finalize;
  object_class->get_property = plugin_ollama_llm_completion_chunk_get_property;
  object_class->set_property = plugin_ollama_llm_completion_chunk_set_property;

  chunk_class->dup_text = plugin_ollama_llm_completion_chunk_dup_text;
  chunk_class->is_done = plugin_ollama_llm_completion_chunk_is_done;

  properties[PROP_NODE] =
    g_param_spec_boxed ("node", NULL, NULL,
                        JSON_TYPE_NODE,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_ollama_llm_completion_chunk_init (PluginOllamaLlmCompletionChunk *self)
{
}

PluginOllamaLlmCompletionChunk *
plugin_ollama_llm_completion_chunk_new (JsonNode *node)
{
  g_return_val_if_fail (node != NULL, NULL);

  return g_object_new (PLUGIN_TYPE_OLLAMA_LLM_COMPLETION_CHUNK,
                       "node", node,
                       NULL);
}
