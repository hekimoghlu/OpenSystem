/* foundry-debugger-instruction.h
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

#include <glib-object.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_INSTRUCTION (foundry_debugger_instruction_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerInstruction, foundry_debugger_instruction, FOUNDRY, DEBUGGER_INSTRUCTION, GObject)

struct _FoundryDebuggerInstructionClass
{
  GObjectClass parent_class;

  guint64  (*get_instruction_pointer)  (FoundryDebuggerInstruction *self);
  char    *(*dup_display_text)         (FoundryDebuggerInstruction *self);
  char    *(*dup_function)             (FoundryDebuggerInstruction *self);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
guint64  foundry_debugger_instruction_get_instruction_pointer (FoundryDebuggerInstruction *self);
FOUNDRY_AVAILABLE_IN_ALL
char    *foundry_debugger_instruction_dup_display_text        (FoundryDebuggerInstruction *self);
FOUNDRY_AVAILABLE_IN_ALL
char    *foundry_debugger_instruction_dup_function            (FoundryDebuggerInstruction *self);

G_END_DECLS
