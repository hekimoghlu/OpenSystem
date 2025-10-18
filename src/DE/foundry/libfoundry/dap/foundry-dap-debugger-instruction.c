/* foundry-dap-debugger-instruction.c
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

#include <stdio.h>

#include "foundry-dap-debugger-instruction-private.h"
#include "foundry-json-node.h"

struct _FoundryDapDebuggerInstruction
{
  FoundryDebuggerInstruction parent_instance;
  JsonNode *node;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerInstruction, foundry_dap_debugger_instruction, FOUNDRY_TYPE_DEBUGGER_INSTRUCTION)

static guint64
foundry_dap_debugger_instruction_get_instruction_pointer (FoundryDebuggerInstruction *instruction)
{
  FoundryDapDebuggerInstruction *self = FOUNDRY_DAP_DEBUGGER_INSTRUCTION (instruction);
  const char *address = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "address", FOUNDRY_JSON_NODE_GET_STRING (&address)))
    {
      guint64 pc = 0;

      if (sscanf (address, "0x%"G_GINT64_MODIFIER"x", &pc) == 1)
        return pc;
    }

  return 0;
}

static char *
foundry_dap_debugger_instruction_dup_display_text (FoundryDebuggerInstruction *instruction)
{
  FoundryDapDebuggerInstruction *self = FOUNDRY_DAP_DEBUGGER_INSTRUCTION (instruction);
  const char *value = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "instruction", FOUNDRY_JSON_NODE_GET_STRING (&value)))
    return g_strdup (value);

  return NULL;
}

static void
foundry_dap_debugger_instruction_finalize (GObject *object)
{
  FoundryDapDebuggerInstruction *self = (FoundryDapDebuggerInstruction *)object;

  g_clear_pointer (&self->node, json_node_unref);

  G_OBJECT_CLASS (foundry_dap_debugger_instruction_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_instruction_class_init (FoundryDapDebuggerInstructionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerInstructionClass *instruction_class = FOUNDRY_DEBUGGER_INSTRUCTION_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_instruction_finalize;

  instruction_class->get_instruction_pointer = foundry_dap_debugger_instruction_get_instruction_pointer;
  instruction_class->dup_display_text = foundry_dap_debugger_instruction_dup_display_text;
}

static void
foundry_dap_debugger_instruction_init (FoundryDapDebuggerInstruction *self)
{
}

FoundryDebuggerInstruction *
foundry_dap_debugger_instruction_new (JsonNode *node)
{
  FoundryDapDebuggerInstruction *self;

  g_return_val_if_fail (node != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_INSTRUCTION, NULL);
  self->node = json_node_ref (node);

  return FOUNDRY_DEBUGGER_INSTRUCTION (self);
}
