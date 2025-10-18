/* foundry-llm-model.c
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

#include "foundry-llm-model.h"
#include "foundry-llm-tool.h"
#include "foundry-util.h"

enum {
  PROP_0,
  PROP_DIGEST,
  PROP_METERED,
  PROP_NAME,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryLlmModel, foundry_llm_model, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_llm_model_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryLlmModel *self = FOUNDRY_LLM_MODEL (object);

  switch (prop_id)
    {
    case PROP_DIGEST:
      g_value_take_string (value, foundry_llm_model_dup_digest (self));
      break;

    case PROP_METERED:
      g_value_set_boolean (value, foundry_llm_model_is_metered (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_llm_model_dup_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_model_class_init (FoundryLlmModelClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_llm_model_get_property;

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DIGEST] =
    g_param_spec_string ("digest", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_METERED] =
    g_param_spec_boolean ("metered", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_llm_model_init (FoundryLlmModel *self)
{
}

char *
foundry_llm_model_dup_name (FoundryLlmModel *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_MODEL (self), NULL);

  if (FOUNDRY_LLM_MODEL_GET_CLASS (self)->dup_name)
    return FOUNDRY_LLM_MODEL_GET_CLASS (self)->dup_name (self);

  return g_strdup (G_OBJECT_TYPE_NAME (self));
}

char *
foundry_llm_model_dup_digest (FoundryLlmModel *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_MODEL (self), NULL);

  if (FOUNDRY_LLM_MODEL_GET_CLASS (self)->dup_digest)
    return FOUNDRY_LLM_MODEL_GET_CLASS (self)->dup_digest (self);

  return NULL;
}

/**
 * foundry_llm_model_complete:
 * @self: a [class@Foundry.LlmModel]
 * @roles: the roles for each message
 * @messages: the message content
 *
 * Requests completion using @roles and @messages.
 *
 * Each element of @roles corresponds to the element at the same index
 * of @messages.
 *
 * The length of @messages and @roles must be the same.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to a [class@Foundry.LlmCompletion].
 */
DexFuture *
foundry_llm_model_complete (FoundryLlmModel    *self,
                            const char * const *roles,
                            const char * const *messages)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_MODEL (self));
  dex_return_error_if_fail (roles != NULL && roles[0] != NULL);
  dex_return_error_if_fail (messages != NULL && messages[0] != NULL);
  dex_return_error_if_fail (g_strv_length ((char **)roles) == g_strv_length ((char **)messages));

  if (FOUNDRY_LLM_MODEL_GET_CLASS (self)->complete)
    return FOUNDRY_LLM_MODEL_GET_CLASS (self)->complete (self, roles, messages);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_llm_model_chat:
 * @self: a [class@Foundry.LlmModel]
 * @system: (nullable): the system prompt for the chat
 *
 * Start a new conversation with the model.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to a [class@Foundry.LlmConversation] or rejects with error.
 */
DexFuture *
foundry_llm_model_chat (FoundryLlmModel *self,
                        const char      *system)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_MODEL (self));

  if (FOUNDRY_LLM_MODEL_GET_CLASS (self)->chat)
    return FOUNDRY_LLM_MODEL_GET_CLASS (self)->chat (self, system);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_llm_model_is_metered:
 * @self: a [class@Foundry.LlmModel]
 *
 * Gets the "metered" property.
 *
 * Returns: %TRUE if the model may be metered.
 */
gboolean
foundry_llm_model_is_metered (FoundryLlmModel *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_MODEL (self), FALSE);

  if (FOUNDRY_LLM_MODEL_GET_CLASS (self)->is_metered)
    return FOUNDRY_LLM_MODEL_GET_CLASS (self)->is_metered (self);

  return FALSE;
}
