/* foundry-on-type-diagnostics.h
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

#include "foundry-text-document.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_ON_TYPE_DIAGNOSTICS (foundry_on_type_diagnostics_get_type())

/**
 * FoundryOnTypeDiagnosticsForeachFunc:
 * @diagnostic: (transfer none): a [class@Foundry.Diagnostic]
 * @user_data: data provided to foreach request
 *
 */
typedef void (*FoundryOnTypeDiagnosticsForeachFunc) (FoundryDiagnostic *diagnostic,
                                                     gpointer           user_data);

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryOnTypeDiagnostics, foundry_on_type_diagnostics, FOUNDRY, ON_TYPE_DIAGNOSTICS, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryOnTypeDiagnostics *foundry_on_type_diagnostics_new              (FoundryTextDocument                 *document);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_on_type_diagnostics_foreach_in_range (FoundryOnTypeDiagnostics            *self,
                                                                        guint                                first_line,
                                                                        guint                                last_line,
                                                                        FoundryOnTypeDiagnosticsForeachFunc  callback,
                                                                        gpointer                             callback_data);

G_END_DECLS
