/* foundry-diagnostic-tool.h
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

#include "foundry-diagnostic-provider.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIAGNOSTIC_TOOL  (foundry_diagnostic_tool_get_type())
#define FOUNDRY_DIAGNOSTIC_TOOL_ERROR (foundry_diagnostic_tool_error_quark())

typedef enum _FoundryDiagnosticToolError
{
  FOUNDRY_DIAGNOSTIC_TOOL_ERROR_NO_COMMAND = 1,
} FoundryDiagnosticToolError;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDiagnosticTool, foundry_diagnostic_tool, FOUNDRY, DIAGNOSTIC_TOOL, FoundryDiagnosticProvider)

struct _FoundryDiagnosticToolClass
{
  FoundryDiagnosticProviderClass parent_class;

  DexFuture *(*prepare)             (FoundryDiagnosticTool  *self,
                                     FoundryProcessLauncher *launcher,
                                     const char * const     *argv,
                                     const char * const     *environ,
                                     const char             *language);
  DexFuture *(*dup_bytes_for_stdin) (FoundryDiagnosticTool  *self,
                                     GFile                  *file,
                                     GBytes                 *contents,
                                     const char             *language);
  DexFuture *(*extract_from_stdout) (FoundryDiagnosticTool  *self,
                                     GFile                  *file,
                                     GBytes                 *contents,
                                     const char             *language,
                                     GBytes                 *stdout_bytes);

  /*< private >*/
  gpointer _reserved[9];
};

FOUNDRY_AVAILABLE_IN_ALL
GQuark   foundry_diagnostic_tool_error_quark (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
char   **foundry_diagnostic_tool_dup_argv    (FoundryDiagnosticTool *self);
FOUNDRY_AVAILABLE_IN_ALL
void     foundry_diagnostic_tool_set_argv    (FoundryDiagnosticTool *self,
                                              const char * const    *argv);
FOUNDRY_AVAILABLE_IN_ALL
char   **foundry_diagnostic_tool_dup_environ (FoundryDiagnosticTool *self);
FOUNDRY_AVAILABLE_IN_ALL
void     foundry_diagnostic_tool_set_environ (FoundryDiagnosticTool *self,
                                              const char * const    *environ);

G_END_DECLS
