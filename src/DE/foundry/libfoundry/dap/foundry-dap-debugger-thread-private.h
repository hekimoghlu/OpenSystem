/* foundry-dap-debugger-thread-private.h
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

#include "foundry-dap-debugger.h"
#include "foundry-debugger-thread.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DAP_DEBUGGER_THREAD (foundry_dap_debugger_thread_get_type())

G_DECLARE_FINAL_TYPE (FoundryDapDebuggerThread, foundry_dap_debugger_thread, FOUNDRY, DAP_DEBUGGER_THREAD, FoundryDebuggerThread)

FoundryDebuggerThread *foundry_dap_debugger_thread_new         (FoundryDapDebugger       *debugger,
                                                                gint64                    id);
gint64                 foundry_dap_debugger_thread_get_id      (FoundryDapDebuggerThread *self);
void                   foundry_dap_debugger_thread_set_stopped (FoundryDapDebuggerThread *self,
                                                                gboolean                  stopped);

G_END_DECLS
