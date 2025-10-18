/* foundry-dap-debugger-variable-private.h
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

#include <json-glib/json-glib.h>

#include "foundry-dap-debugger.h"
#include "foundry-debugger-variable.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DAP_DEBUGGER_VARIABLE (foundry_dap_debugger_variable_get_type())

G_DECLARE_FINAL_TYPE (FoundryDapDebuggerVariable, foundry_dap_debugger_variable, FOUNDRY, DAP_DEBUGGER_VARIABLE, FoundryDebuggerVariable)

FoundryDebuggerVariable *foundry_dap_debugger_variable_new (FoundryDapDebugger *debugger,
                                                            JsonNode           *node);

G_END_DECLS
