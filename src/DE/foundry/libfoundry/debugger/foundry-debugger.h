/* foundry-debugger.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER (foundry_debugger_get_type())
#define FOUNDRY_TYPE_DEBUGGER_MOVEMENT (foundry_debugger_movement_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebugger, foundry_debugger, FOUNDRY, DEBUGGER, FoundryContextual)

struct _FoundryDebuggerClass
{
  FoundryContextualClass parent_class;

  char                  *(*dup_name)           (FoundryDebugger           *self);
  DexFuture             *(*initialize)         (FoundryDebugger           *self);
  DexFuture             *(*connect_to_target)  (FoundryDebugger           *self,
                                                FoundryDebuggerTarget     *target);
  GListModel            *(*list_address_space) (FoundryDebugger           *self);
  GListModel            *(*list_modules)       (FoundryDebugger           *self);
  GListModel            *(*list_traps)         (FoundryDebugger           *self);
  GListModel            *(*list_threads)       (FoundryDebugger           *self);
  GListModel            *(*list_thread_groups) (FoundryDebugger           *self);
  DexFuture             *(*disassemble)        (FoundryDebugger           *self,
                                                guint64                    begin_address,
                                                guint64                    end_address);
  DexFuture             *(*interpret)          (FoundryDebugger           *self,
                                                const char                *command);
  DexFuture             *(*interrupt)          (FoundryDebugger           *self);
  DexFuture             *(*stop)               (FoundryDebugger           *self);
  DexFuture             *(*send_signal)        (FoundryDebugger           *self,
                                                int                        signum);
  DexFuture             *(*move)               (FoundryDebugger           *self,
                                                FoundryDebuggerMovement    movement);
  DexFuture             *(*trap)               (FoundryDebugger           *self,
                                                FoundryDebuggerTrapParams *params);
  gboolean               (*can_move)           (FoundryDebugger           *self,
                                                FoundryDebuggerMovement    movement);
  void                   (*event)              (FoundryDebugger           *self,
                                                FoundryDebuggerEvent      *event);
  GListModel            *(*list_log_messages)  (FoundryDebugger           *self);
  FoundryDebuggerThread *(*dup_primary_thread) (FoundryDebugger           *self);
  gboolean               (*has_terminated)     (FoundryDebugger           *self);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
GType                  foundry_debugger_movement_get_type  (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_debugger_dup_name           (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_initialize         (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_connect_to_target  (FoundryDebugger           *self,
                                                            FoundryDebuggerTarget     *target);
FOUNDRY_AVAILABLE_IN_ALL
GListModel            *foundry_debugger_list_address_space (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel            *foundry_debugger_list_modules       (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel            *foundry_debugger_list_traps         (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel            *foundry_debugger_list_threads       (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel            *foundry_debugger_list_thread_groups (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_disassemble        (FoundryDebugger           *self,
                                                            guint64                    begin_address,
                                                            guint64                    end_address);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_interpret          (FoundryDebugger           *self,
                                                            const char                *command);
FOUNDRY_DEPRECATED_IN_1_1_FOR(foundry_debugger_thread_interrupt)
DexFuture             *foundry_debugger_interrupt          (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_send_signal        (FoundryDebugger           *self,
                                                            int                        signum);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_stop               (FoundryDebugger           *self);
FOUNDRY_DEPRECATED_IN_1_1_FOR(foundry_debugger_thread_can_move)
gboolean               foundry_debugger_can_move           (FoundryDebugger           *self,
                                                            FoundryDebuggerMovement    movement);
FOUNDRY_DEPRECATED_IN_1_1_FOR(foundry_debugger_thread_move)
DexFuture             *foundry_debugger_move               (FoundryDebugger           *self,
                                                            FoundryDebuggerMovement    movement);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_debugger_trap               (FoundryDebugger           *self,
                                                            FoundryDebuggerTrapParams *params);
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_debugger_emit_event         (FoundryDebugger           *self,
                                                            FoundryDebuggerEvent      *event);
FOUNDRY_AVAILABLE_IN_1_1
GListModel            *foundry_debugger_list_log_messages  (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_1_1
FoundryDebuggerThread *foundry_debugger_dup_primary_thread (FoundryDebugger           *self);
FOUNDRY_AVAILABLE_IN_1_1
gboolean               foundry_debugger_has_terminated     (FoundryDebugger           *self);

G_END_DECLS
