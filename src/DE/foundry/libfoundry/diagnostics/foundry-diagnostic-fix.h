/* foundry-diagnostic-fix.h
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

#define FOUNDRY_TYPE_DIAGNOSTIC_FIX (foundry_diagnostic_fix_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_FINAL_TYPE (FoundryDiagnosticFix, foundry_diagnostic_fix, FOUNDRY, DIAGNOSTIC_FIX, GObject)

FOUNDRY_AVAILABLE_IN_1_1
char       *foundry_diagnostic_fix_dup_description (FoundryDiagnosticFix *self);
FOUNDRY_AVAILABLE_IN_1_1
GListModel *foundry_diagnostic_fix_list_text_edits (FoundryDiagnosticFix *self);

G_END_DECLS
