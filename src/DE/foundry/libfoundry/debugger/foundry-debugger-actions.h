/* foundry-debugger-actions.h
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

#define FOUNDRY_TYPE_DEBUGGER_ACTIONS (foundry_debugger_actions_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_FINAL_TYPE (FoundryDebuggerActions, foundry_debugger_actions, FOUNDRY, DEBUGGER_ACTIONS, GObject)

FOUNDRY_AVAILABLE_IN_1_1
FoundryDebuggerActions *foundry_debugger_actions_new          (FoundryDebugger        *debugger,
                                                               FoundryDebuggerThread  *thread);
FOUNDRY_AVAILABLE_IN_1_1
FoundryDebugger        *foundry_debugger_actions_dup_debugger (FoundryDebuggerActions *self);
FOUNDRY_AVAILABLE_IN_1_1
void                    foundry_debugger_actions_set_debugger (FoundryDebuggerActions *self,
                                                               FoundryDebugger        *debugger);
FOUNDRY_AVAILABLE_IN_1_1
FoundryDebuggerThread  *foundry_debugger_actions_dup_thread   (FoundryDebuggerActions *self);
FOUNDRY_AVAILABLE_IN_1_1
void                    foundry_debugger_actions_set_thread   (FoundryDebuggerActions *self,
                                                               FoundryDebuggerThread  *thread);

G_END_DECLS
