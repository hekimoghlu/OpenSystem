/* foundry-debugger-trap-params.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_TRAP_PARAMS      (foundry_debugger_trap_params_get_type())
#define FOUNDRY_TYPE_DEBUGGER_TRAP_DISPOSITION (foundry_debugger_trap_disposition_get_type())
#define FOUNDRY_TYPE_DEBUGGER_TRAP_KIND        (foundry_debugger_trap_kind_get_type())
#define FOUNDRY_TYPE_DEBUGGER_WATCH_ACCESS     (foundry_debugger_watch_access_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDebuggerTrapParams, foundry_debugger_trap_params, FOUNDRY, DEBUGGER_TRAP_PARAMS, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GType                           foundry_debugger_trap_disposition_get_type           (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
GType                           foundry_debugger_trap_kind_get_type                  (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
GType                           foundry_debugger_watch_access_get_type               (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_1_1
FoundryDebuggerTrapParams      *foundry_debugger_trap_params_new                     (void);
FOUNDRY_AVAILABLE_IN_1_1
FoundryDebuggerTrapParams      *foundry_debugger_trap_params_copy                    (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
char                           *foundry_debugger_trap_params_dup_function            (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_function            (FoundryDebuggerTrapParams      *self,
                                                                                      const char                     *function);
FOUNDRY_AVAILABLE_IN_ALL
char                           *foundry_debugger_trap_params_dup_thread_id           (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_thread_id           (FoundryDebuggerTrapParams      *self,
                                                                                      const char                     *thread_id);
FOUNDRY_AVAILABLE_IN_ALL
char                           *foundry_debugger_trap_params_dup_stack_frame_id      (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_stack_frame_id      (FoundryDebuggerTrapParams      *self,
                                                                                      const char                     *stack_frame_id);
FOUNDRY_AVAILABLE_IN_ALL
char                           *foundry_debugger_trap_params_dup_path                (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_path                (FoundryDebuggerTrapParams      *self,
                                                                                      const char                     *path);
FOUNDRY_AVAILABLE_IN_ALL
guint                           foundry_debugger_trap_params_get_line                (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_line                (FoundryDebuggerTrapParams      *self,
                                                                                      guint                           line);
FOUNDRY_AVAILABLE_IN_ALL
guint                           foundry_debugger_trap_params_get_line_offset         (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_line_offset         (FoundryDebuggerTrapParams      *self,
                                                                                      guint                           line_offset);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerTrapDisposition  foundry_debugger_trap_params_get_disposition         (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_disposition         (FoundryDebuggerTrapParams      *self,
                                                                                      FoundryDebuggerTrapDisposition  disposition);
FOUNDRY_AVAILABLE_IN_ALL
guint64                         foundry_debugger_trap_params_get_instruction_pointer (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_instruction_pointer (FoundryDebuggerTrapParams      *self,
                                                                                      guint64                         instruction_pointer);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerTrapKind         foundry_debugger_trap_params_get_kind                (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_kind                (FoundryDebuggerTrapParams      *self,
                                                                                      FoundryDebuggerTrapKind         kind);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerWatchAccess      foundry_debugger_trap_params_get_access              (FoundryDebuggerTrapParams      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                            foundry_debugger_trap_params_set_access              (FoundryDebuggerTrapParams      *self,
                                                                                      FoundryDebuggerWatchAccess      access);

G_END_DECLS
