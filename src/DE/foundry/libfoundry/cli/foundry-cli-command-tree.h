/* foundry-cli-command-tree.h
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

#include <glib-object.h>

#include "foundry-cli-command.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_CLI_COMMAND_TREE (foundry_cli_command_tree_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryCliCommandTree, foundry_cli_command_tree, FOUNDRY, CLI_COMMAND_TREE, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryCliCommandTree    *foundry_cli_command_tree_get_default (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCliCommandTree    *foundry_cli_command_tree_new         (void);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_cli_command_tree_register    (FoundryCliCommandTree     *self,
                                                                const char * const        *path,
                                                                const FoundryCliCommand   *command);
FOUNDRY_AVAILABLE_IN_ALL
const FoundryCliCommand  *foundry_cli_command_tree_lookup      (FoundryCliCommandTree     *self,
                                                                char                    ***args,
                                                                FoundryCliOptions        **options,
                                                                GError                   **error);
FOUNDRY_AVAILABLE_IN_ALL
char                    **foundry_cli_command_tree_complete    (FoundryCliCommandTree     *self,
                                                                FoundryCommandLine        *command_line,
                                                                const char                *line,
                                                                int                        point,
                                                                const char                *current);

G_END_DECLS
