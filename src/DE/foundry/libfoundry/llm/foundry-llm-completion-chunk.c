/* foundry-llm-completion-chunk.c
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

#include "foundry-llm-completion-chunk.h"

enum {
  PROP_0,
  PROP_IS_DONE,
  PROP_TEXT,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryLlmCompletionChunk, foundry_llm_completion_chunk, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_llm_completion_chunk_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryLlmCompletionChunk *self = FOUNDRY_LLM_COMPLETION_CHUNK (object);

  switch (prop_id)
    {
    case PROP_IS_DONE:
      g_value_set_boolean (value, foundry_llm_completion_chunk_is_done (self));
      break;

    case PROP_TEXT:
      g_value_take_string (value, foundry_llm_completion_chunk_dup_text (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_completion_chunk_class_init (FoundryLlmCompletionChunkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_llm_completion_chunk_get_property;

  properties[PROP_IS_DONE] =
    g_param_spec_boolean ("is-done", NULL, NULL,
                          TRUE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_TEXT] =
    g_param_spec_string ("text", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_llm_completion_chunk_init (FoundryLlmCompletionChunk *self)
{
}

/**
 * foundry_llm_completion_chunk_dup_text:
 * @self: a [class@Foundry.LlmCompletionChunk]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_llm_completion_chunk_dup_text (FoundryLlmCompletionChunk *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_COMPLETION_CHUNK (self), NULL);

  if (FOUNDRY_LLM_COMPLETION_CHUNK_GET_CLASS (self)->dup_text)
    return FOUNDRY_LLM_COMPLETION_CHUNK_GET_CLASS (self)->dup_text (self);

  return NULL;
}

/**
 * foundry_llm_completion_chunk_is_done:
 * @self: a [class@Foundry.LlmCompletionChunk]
 *
 * Returns: %TRUE if this is the last chunk
 */
gboolean
foundry_llm_completion_chunk_is_done (FoundryLlmCompletionChunk *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_COMPLETION_CHUNK (self), FALSE);

  if (FOUNDRY_LLM_COMPLETION_CHUNK_GET_CLASS (self)->is_done)
    return FOUNDRY_LLM_COMPLETION_CHUNK_GET_CLASS (self)->is_done (self);

  return TRUE;
}
