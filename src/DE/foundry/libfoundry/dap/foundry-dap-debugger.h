/* foundry-dap-debugger.h
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

#include "foundry-debugger.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DAP_DEBUGGER       (foundry_dap_debugger_get_type())
#define FOUNDRY_TYPE_DAP_DEBUGGER_QUIRK (foundry_dap_debugger_quirk_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_DERIVABLE_TYPE (FoundryDapDebugger, foundry_dap_debugger, FOUNDRY, DAP_DEBUGGER, FoundryDebugger)

struct _FoundryDapDebuggerClass
{
  FoundryDebuggerClass parent_class;

  /*< private >*/
  gpointer _reserved[8];
};

typedef enum _FoundryDapDebuggerQuirk
{
  FOUNDRY_DAP_DEBUGGER_QUIRK_NONE          = 0,
  FOUNDRY_DAP_DEBUGGER_QUIRK_QUERY_THREADS = 1 << 0,
} FoundryDapDebuggerQuirk;

FOUNDRY_AVAILABLE_IN_1_1
GType                    foundry_dap_debugger_quirk_get_type (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_1_1
GSubprocess             *foundry_dap_debugger_dup_subprocess (FoundryDapDebugger *self);
FOUNDRY_AVAILABLE_IN_1_1
GIOStream               *foundry_dap_debugger_dup_stream     (FoundryDapDebugger *self);
FOUNDRY_AVAILABLE_IN_1_1
DexFuture               *foundry_dap_debugger_call           (FoundryDapDebugger *self,
                                                              JsonNode           *node) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
DexFuture               *foundry_dap_debugger_send           (FoundryDapDebugger *self,
                                                              JsonNode           *node) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
FoundryDapDebuggerQuirk  foundry_dap_debugger_get_quirks     (FoundryDapDebugger *self);

G_END_DECLS
