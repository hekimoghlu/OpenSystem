/* foundry-llm-message.h
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

#include <gio/gio.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LLM_MESSAGE (foundry_llm_message_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLlmMessage, foundry_llm_message, FOUNDRY, LLM_MESSAGE, GObject)

struct _FoundryLlmMessageClass
{
  GObjectClass parent_class;

  char       *(*dup_content)     (FoundryLlmMessage *self);
  char       *(*dup_role)        (FoundryLlmMessage *self);
  gboolean    (*has_tool_call)   (FoundryLlmMessage *self);
  GListModel *(*list_tool_calls) (FoundryLlmMessage *self);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_llm_message_dup_content     (FoundryLlmMessage *self);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_llm_message_dup_role        (FoundryLlmMessage *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean    foundry_llm_message_has_tool_call   (FoundryLlmMessage *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel *foundry_llm_message_list_tool_calls (FoundryLlmMessage *self);

G_END_DECLS
