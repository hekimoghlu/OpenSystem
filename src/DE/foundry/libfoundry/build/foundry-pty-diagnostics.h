/* foundry-pty-diagnostics.h
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

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PTY_DIAGNOSTICS (foundry_pty_diagnostics_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryPtyDiagnostics, foundry_pty_diagnostics, FOUNDRY, PTY_DIAGNOSTICS, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_pty_diagnostics_register        (GRegex                 *regex);
FOUNDRY_AVAILABLE_IN_ALL
FoundryPtyDiagnostics *foundry_pty_diagnostics_new             (FoundryContext         *context,
                                                                int                     pty_fd);
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_pty_diagnostics_reset           (FoundryPtyDiagnostics  *self);
FOUNDRY_AVAILABLE_IN_ALL
int                    foundry_pty_diagnostics_get_fd          (FoundryPtyDiagnostics  *self);
FOUNDRY_AVAILABLE_IN_ALL
int                    foundry_pty_diagnostics_create_producer (FoundryPtyDiagnostics  *self,
                                                                GError                **error);

G_END_DECLS
