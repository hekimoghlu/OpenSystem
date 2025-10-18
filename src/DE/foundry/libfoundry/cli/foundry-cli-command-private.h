/* foundry-cli-command-private.h
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

#include "foundry-cli-command.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command-line.h"

G_BEGIN_DECLS

DexFuture         *foundry_cli_command_run  (const FoundryCliCommand *command,
                                             FoundryCommandLine      *command_line,
                                             const char * const      *argv,
                                             FoundryCliOptions       *options,
                                             DexCancellable          *cancellable);
FoundryCliCommand *foundry_cli_command_copy (const FoundryCliCommand *command);
void               foundry_cli_command_free (FoundryCliCommand       *command);

G_END_DECLS
