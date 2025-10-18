/* foundry-debugger-stack-frame.h
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

#define FOUNDRY_TYPE_DEBUGGER_STACK_FRAME (foundry_debugger_stack_frame_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerStackFrame, foundry_debugger_stack_frame, FOUNDRY, DEBUGGER_STACK_FRAME, GObject)

struct _FoundryDebuggerStackFrameClass
{
  GObjectClass parent_class;

  char                  *(*dup_id)                  (FoundryDebuggerStackFrame *self);
  char                  *(*dup_name)                (FoundryDebuggerStackFrame *self);
  char                  *(*dup_module_id)           (FoundryDebuggerStackFrame *self);
  guint64                (*get_instruction_pointer) (FoundryDebuggerStackFrame *self);
  gboolean               (*can_restart)             (FoundryDebuggerStackFrame *self);
  FoundryDebuggerSource *(*dup_source)              (FoundryDebuggerStackFrame *self);
  void                   (*get_source_range)        (FoundryDebuggerStackFrame *self,
                                                     guint                     *begin_line,
                                                     guint                     *begin_line_offset,
                                                     guint                     *end_line,
                                                     guint                     *end_line_offset);
  DexFuture             *(*list_params)             (FoundryDebuggerStackFrame *self);
  DexFuture             *(*list_locals)             (FoundryDebuggerStackFrame *self);
  DexFuture             *(*list_registers)          (FoundryDebuggerStackFrame *self);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_debugger_stack_frame_dup_id                  (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_debugger_stack_frame_dup_name                (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_debugger_stack_frame_dup_module_id           (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
guint64                foundry_debugger_stack_frame_get_instruction_pointer (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean               foundry_debugger_stack_frame_can_restart             (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_debugger_stack_frame_get_source_range        (FoundryDebuggerStackFrame  *self,
                                                                             guint                      *begin_line,
                                                                             guint                      *begin_line_offset,
                                                                             guint                      *end_line,
                                                                             guint                      *end_line_offset);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerSource *foundry_debugger_stack_frame_dup_source              (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_stack_frame_list_params             (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_stack_frame_list_locals             (FoundryDebuggerStackFrame  *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_stack_frame_list_registers          (FoundryDebuggerStackFrame  *self);

G_END_DECLS
