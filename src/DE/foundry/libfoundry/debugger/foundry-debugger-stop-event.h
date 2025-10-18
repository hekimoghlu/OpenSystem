/* foundry-debugger-stop-event.h
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

#include "foundry-debugger-event.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_STOP_EVENT  (foundry_debugger_stop_event_get_type())
#define FOUNDRY_TYPE_DEBUGGER_STOP_REASON (foundry_debugger_stop_reason_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerStopEvent, foundry_debugger_stop_event, FOUNDRY, DEBUGGER_STOP_EVENT, FoundryDebuggerEvent)

struct _FoundryDebuggerStopEventClass
{
  FoundryDebuggerEventClass parent_class;

  FoundryDebuggerStopReason  (*get_reason)    (FoundryDebuggerStopEvent *self);
  FoundryDebuggerTrap       *(*dup_trap)      (FoundryDebuggerStopEvent *self);
  int                        (*get_signal)    (FoundryDebuggerStopEvent *self);
  int                        (*get_exit_code) (FoundryDebuggerStopEvent *self);

  /*< private >*/
  gpointer _reserved[4];
};

FOUNDRY_AVAILABLE_IN_ALL
GType                      foundry_debugger_stop_reason_get_type     (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerStopReason  foundry_debugger_stop_event_get_reason    (FoundryDebuggerStopEvent *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerTrap       *foundry_debugger_stop_event_dup_trap      (FoundryDebuggerStopEvent *self);
FOUNDRY_AVAILABLE_IN_ALL
int                        foundry_debugger_stop_event_get_signal    (FoundryDebuggerStopEvent *self);
FOUNDRY_AVAILABLE_IN_ALL
int                        foundry_debugger_stop_event_get_exit_code (FoundryDebuggerStopEvent *self);

G_END_DECLS
