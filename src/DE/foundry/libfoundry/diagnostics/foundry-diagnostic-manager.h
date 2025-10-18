/* foundry-diagnostic-manager.h
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

#include "foundry-service.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIAGNOSTIC_MANAGER (foundry_diagnostic_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryDiagnosticManager, foundry_diagnostic_manager, FOUNDRY, DIAGNOSTIC_MANAGER, FoundryService)

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_diagnostic_manager_diagnose       (FoundryDiagnosticManager  *self,
                                                      GFile                     *file,
                                                      GBytes                    *contents,
                                                      const char                *language) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_diagnostic_manager_diagnose_file  (FoundryDiagnosticManager  *self,
                                                      GFile                     *file) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_diagnostic_manager_diagnose_files (FoundryDiagnosticManager  *self,
                                                      GFile                    **files,
                                                      guint                      n_files) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
DexFuture *foundry_diagnostic_manager_list_all       (FoundryDiagnosticManager  *self) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
