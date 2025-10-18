/* foundry-llm-conversation.h
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

#pragma once

#include <libdex.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LLM_CONVERSATION (foundry_llm_conversation_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLlmConversation, foundry_llm_conversation, FOUNDRY, LLM_CONVERSATION, GObject)

struct _FoundryLlmConversationClass
{
  GObjectClass parent_class;

  DexFuture  *(*add_context)   (FoundryLlmConversation *self,
                                const char             *context);
  DexFuture  *(*send_messages) (FoundryLlmConversation *self,
                                const char * const     *roles,
                                const char * const     *messages);
  void        (*reset)         (FoundryLlmConversation *self);
  GListModel *(*list_history)  (FoundryLlmConversation *self);
  DexFuture  *(*call)          (FoundryLlmConversation *self,
                                FoundryLlmToolCall     *call);
  gboolean    (*is_busy)       (FoundryLlmConversation *self);

  /*< private >*/
  gpointer _reserved[9];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_llm_conversation_add_context   (FoundryLlmConversation *self,
                                                    const char             *context);
FOUNDRY_AVAILABLE_IN_ALL
void        foundry_llm_conversation_set_tools     (FoundryLlmConversation *self,
                                                    GListModel             *tools);
FOUNDRY_AVAILABLE_IN_ALL
GListModel *foundry_llm_conversation_dup_tools     (FoundryLlmConversation *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_llm_conversation_send_message  (FoundryLlmConversation *self,
                                                    const char             *role,
                                                    const char             *message);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_llm_conversation_send_messages (FoundryLlmConversation *self,
                                                    const char * const     *roles,
                                                    const char * const     *messages);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_llm_conversation_call          (FoundryLlmConversation *self,
                                                    FoundryLlmToolCall     *call);
FOUNDRY_AVAILABLE_IN_ALL
void        foundry_llm_conversation_reset         (FoundryLlmConversation *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel *foundry_llm_conversation_list_history  (FoundryLlmConversation *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean    foundry_llm_conversation_is_busy       (FoundryLlmConversation *self);

G_END_DECLS
