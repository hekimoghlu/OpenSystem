/* foundry-simple-llm-message.c
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

#include "foundry-simple-llm-message.h"

struct _FoundrySimpleLlmMessage
{
  FoundryLlmMessage parent_instance;
  char *role;
  char *content;
};

G_DEFINE_FINAL_TYPE (FoundrySimpleLlmMessage, foundry_simple_llm_message, FOUNDRY_TYPE_LLM_MESSAGE)

static char *
foundry_simple_llm_message_dup_role (FoundryLlmMessage *message)
{
  return g_strdup (FOUNDRY_SIMPLE_LLM_MESSAGE (message)->role);
}

static char *
foundry_simple_llm_message_dup_content (FoundryLlmMessage *message)
{
  return g_strdup (FOUNDRY_SIMPLE_LLM_MESSAGE (message)->content);
}

static void
foundry_simple_llm_message_finalize (GObject *object)
{
  FoundrySimpleLlmMessage *self = (FoundrySimpleLlmMessage *)object;

  g_clear_pointer (&self->role, g_free);
  g_clear_pointer (&self->content, g_free);

  G_OBJECT_CLASS (foundry_simple_llm_message_parent_class)->finalize (object);
}

static void
foundry_simple_llm_message_class_init (FoundrySimpleLlmMessageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryLlmMessageClass *llm_message_class = FOUNDRY_LLM_MESSAGE_CLASS (klass);

  object_class->finalize = foundry_simple_llm_message_finalize;

  llm_message_class->dup_role = foundry_simple_llm_message_dup_role;
  llm_message_class->dup_content = foundry_simple_llm_message_dup_content;
}

static void
foundry_simple_llm_message_init (FoundrySimpleLlmMessage *self)
{
}

/**
 * foundry_simple_llm_message_new:
 * @role: (transfer full):
 * @content: (transfer full):
 *
 * Returns: (transfer full):
 */
FoundryLlmMessage *
foundry_simple_llm_message_new (char *role,
                                char *content)
{
  FoundrySimpleLlmMessage *self;

  g_return_val_if_fail (role != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_SIMPLE_LLM_MESSAGE, NULL);
  self->role = role;
  self->content = content;

  return FOUNDRY_LLM_MESSAGE (self);
}
