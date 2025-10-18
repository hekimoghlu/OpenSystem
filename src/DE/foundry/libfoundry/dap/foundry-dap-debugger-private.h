/* foundry-dap-debugger-private.h
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

#include "foundry-dap-debugger.h"

G_BEGIN_DECLS

DexFuture *_foundry_dap_debugger_move              (FoundryDapDebugger      *self,
                                                    gint64                   thread_id,
                                                    FoundryDebuggerMovement  movement);
DexFuture *_foundry_dap_debugger_remove_breakpoint (FoundryDapDebugger      *self,
                                                    gint64                   breakpoint_id);

G_END_DECLS
