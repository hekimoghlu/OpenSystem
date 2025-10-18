/* foundry-terminal-launcher.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TERMINAL_LAUNCHER (foundry_terminal_launcher_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryTerminalLauncher, foundry_terminal_launcher, FOUNDRY, TERMINAL_LAUNCHER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryTerminalLauncher  *foundry_terminal_launcher_new                      (FoundryCommand          *command,
                                                                              const char * const      *override_environment);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTerminalLauncher  *foundry_terminal_launcher_copy                     (FoundryTerminalLauncher *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                *foundry_terminal_launcher_run                      (FoundryTerminalLauncher *self,
                                                                              int                      pty_fd);
FOUNDRY_AVAILABLE_IN_ALL
char                    **foundry_terminal_launcher_dup_override_environment (FoundryTerminalLauncher *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommand           *foundry_terminal_launcher_dup_command              (FoundryTerminalLauncher *self);

G_END_DECLS
