/* foundry-diagnostic.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIAGNOSTIC          (foundry_diagnostic_get_type())
#define FOUNDRY_TYPE_DIAGNOSTIC_SEVERITY (foundry_diagnostic_severity_get_type())

typedef enum _FoundryDiagnosticSeverity
{
  FOUNDRY_DIAGNOSTIC_IGNORED    = 0,
  FOUNDRY_DIAGNOSTIC_NOTE       = 1,
  FOUNDRY_DIAGNOSTIC_UNUSED     = 2,
  FOUNDRY_DIAGNOSTIC_DEPRECATED = 3,
  FOUNDRY_DIAGNOSTIC_WARNING    = 4,
  FOUNDRY_DIAGNOSTIC_ERROR      = 5,
  FOUNDRY_DIAGNOSTIC_FATAL      = 6,
} FoundryDiagnosticSeverity;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDiagnostic, foundry_diagnostic, FOUNDRY, DIAGNOSTIC, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GType                      foundry_diagnostic_severity_get_type (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
gboolean                   foundry_diagnostic_equal             (const FoundryDiagnostic *left,
                                                                 const FoundryDiagnostic *right);
FOUNDRY_AVAILABLE_IN_ALL
int                        foundry_diagnostic_compare           (FoundryDiagnostic       *left,
                                                                 FoundryDiagnostic       *right);
FOUNDRY_AVAILABLE_IN_ALL
guint                      foundry_diagnostic_hash              (gconstpointer            data);
FOUNDRY_AVAILABLE_IN_ALL
GFile                     *foundry_diagnostic_dup_file          (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
guint                      foundry_diagnostic_get_line          (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
guint                      foundry_diagnostic_get_line_offset   (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryMarkup             *foundry_diagnostic_dup_markup        (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_diagnostic_dup_message       (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_diagnostic_dup_path          (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_1_1
char                      *foundry_diagnostic_dup_rule_id       (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDiagnosticSeverity  foundry_diagnostic_get_severity      (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel                *foundry_diagnostic_list_ranges       (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_1_1
GListModel                *foundry_diagnostic_list_fixes        (FoundryDiagnostic       *self);
FOUNDRY_AVAILABLE_IN_1_1
gboolean                   foundry_diagnostic_has_fix           (FoundryDiagnostic       *self);

G_END_DECLS
