/* foundry-llm-tool-call.h
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

#pragma once

#include <libdex.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LLM_TOOL_CALL (foundry_llm_tool_call_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLlmToolCall, foundry_llm_tool_call, FOUNDRY, LLM_TOOL_CALL, GObject)

struct _FoundryLlmToolCallClass
{
  GObjectClass parent_class;

  char      *(*dup_title)    (FoundryLlmToolCall *self);
  char      *(*dup_subtitle) (FoundryLlmToolCall *self);
  gboolean   (*is_callable)  (FoundryLlmToolCall *self);
  DexFuture *(*confirm)      (FoundryLlmToolCall *self);
  DexFuture *(*deny)         (FoundryLlmToolCall *self);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_llm_tool_call_dup_title    (FoundryLlmToolCall *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_llm_tool_call_dup_subtitle (FoundryLlmToolCall *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_llm_tool_call_confirm      (FoundryLlmToolCall *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_llm_tool_call_deny         (FoundryLlmToolCall *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_llm_tool_call_is_callable  (FoundryLlmToolCall *self);

G_END_DECLS
