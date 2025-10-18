/* foundry-simple-llm-message.h
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

#include "foundry-llm-message.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SIMPLE_LLM_MESSAGE (foundry_simple_llm_message_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundrySimpleLlmMessage, foundry_simple_llm_message, FOUNDRY, SIMPLE_LLM_MESSAGE, FoundryLlmMessage)

FOUNDRY_AVAILABLE_IN_ALL
FoundryLlmMessage *foundry_simple_llm_message_new (char *role,
                                                   char *content);

G_END_DECLS
