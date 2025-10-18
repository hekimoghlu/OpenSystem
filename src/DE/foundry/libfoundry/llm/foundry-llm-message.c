/* foundry-llm-message.c
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

#include "foundry-llm-message.h"

G_DEFINE_ABSTRACT_TYPE (FoundryLlmMessage, foundry_llm_message, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_CONTENT,
  PROP_ROLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_llm_message_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundryLlmMessage *self = FOUNDRY_LLM_MESSAGE (object);

  switch (prop_id)
    {
    case PROP_CONTENT:
      g_value_take_string (value, foundry_llm_message_dup_content (self));
      break;

    case PROP_ROLE:
      g_value_take_string (value, foundry_llm_message_dup_role (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_message_class_init (FoundryLlmMessageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_llm_message_get_property;

  properties[PROP_CONTENT] =
    g_param_spec_string ("content", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ROLE] =
    g_param_spec_string ("role", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_llm_message_init (FoundryLlmMessage *self)
{
}

/**
 * foundry_llm_message_dup_role:
 * @self: a [class@Foundry.LlmMessage]
 *
 * The role of the message such as "system", "user", "assistant", or "tool".
 *
 * Returns: (transfer full): the role as a string
 */
char *
foundry_llm_message_dup_role (FoundryLlmMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_MESSAGE (self), NULL);

  if (FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->dup_role)
   return FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->dup_role (self);

  return NULL;
}

/**
 * foundry_llm_message_dup_content:
 * @self: a [class@Foundry.LlmMessage]
 *
 * The content of the message.
 *
 * Returns: (transfer full) (not nullable):
 */
char *
foundry_llm_message_dup_content (FoundryLlmMessage *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_LLM_MESSAGE (self), NULL);

  if (FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->dup_content)
    ret = FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->dup_content (self);

  return ret ? ret : g_strdup ("");
}

gboolean
foundry_llm_message_has_tool_call (FoundryLlmMessage *self)
{
  g_autoptr(GListModel) tool_calls = NULL;

  g_return_val_if_fail (FOUNDRY_IS_LLM_MESSAGE (self), FALSE);

  if (FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->has_tool_call)
    return FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->has_tool_call (self);

  tool_calls = foundry_llm_message_list_tool_calls (self);

  return tool_calls != NULL &&
         g_list_model_get_n_items (tool_calls) > 0;
}

/**
 * foundry_llm_message_list_tool_calls:
 * @self: a [class@Foundry.LlmMessage]
 *
 * Returns: (transfer full) (nullable): a [iface@Gio.ListModel] of
 *   [class@Foundry.LlmToolCall] or %NULL
 */
GListModel *
foundry_llm_message_list_tool_calls (FoundryLlmMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_MESSAGE (self), NULL);

  if (FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->list_tool_calls)
    return FOUNDRY_LLM_MESSAGE_GET_CLASS (self)->list_tool_calls (self);

  return NULL;
}
