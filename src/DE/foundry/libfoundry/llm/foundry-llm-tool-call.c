/* foundry-llm-tool-call.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-llm-tool-call.h"
#include "foundry-simple-llm-message.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryLlmToolCall, foundry_llm_tool_call, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_IS_CALLABLE,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_llm_tool_call_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryLlmToolCall *self = FOUNDRY_LLM_TOOL_CALL (object);

  switch (prop_id)
    {
    case PROP_IS_CALLABLE:
      g_value_set_boolean (value, foundry_llm_tool_call_is_callable (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_llm_tool_call_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_llm_tool_call_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_tool_call_class_init (FoundryLlmToolCallClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_llm_tool_call_get_property;

  properties[PROP_IS_CALLABLE] =
    g_param_spec_boolean ("is-callable", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_llm_tool_call_init (FoundryLlmToolCall *self)
{
}

char *
foundry_llm_tool_call_dup_subtitle (FoundryLlmToolCall *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL_CALL (self), NULL);

  if (FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->dup_subtitle)
    return FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->dup_subtitle (self);

  return NULL;
}

char *
foundry_llm_tool_call_dup_title (FoundryLlmToolCall *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL_CALL (self), NULL);

  if (FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->dup_title)
    return FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->dup_title (self);

  return NULL;
}

gboolean
foundry_llm_tool_call_is_callable (FoundryLlmToolCall *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL_CALL (self), FALSE);

  if (FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->is_callable)
    return FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->is_callable (self);

  return FALSE;
}

/**
 * foundry_llm_tool_call_confirm:
 * @self: a [class@Foundry.LlmToolCall]
 *
 * Confirms that the tool call is authorized and performs it.
 *
 * The result should be a [class@Foundry.LlmMessage] that may be sent
 * back to the model to continue the conversation.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.LlmMessage] or rejects with error.
 */
DexFuture *
foundry_llm_tool_call_confirm (FoundryLlmToolCall *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_TOOL_CALL (self));

  if (FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->confirm)
    return FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->confirm (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_llm_tool_call_deny:
 * @self: a [class@Foundry.LlmToolCall]
 *
 * Denies that the tool call is authorized and returns a message of
 * failure to be sent back to the model.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.LlmMessage] or rejects with error.
 */
DexFuture *
foundry_llm_tool_call_deny (FoundryLlmToolCall *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_TOOL_CALL (self));

  if (FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->deny)
    return FOUNDRY_LLM_TOOL_CALL_GET_CLASS (self)->deny (self);

  return dex_future_new_take_object (foundry_simple_llm_message_new (g_strdup ("user"),
                                                                     g_strdup ("The command was not authorized by the user.")));
}
