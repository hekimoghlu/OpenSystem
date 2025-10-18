/* foundry-run-tool-private.h
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

#include "foundry-run-tool.h"

G_BEGIN_DECLS

void       foundry_run_tool_set_subprocess (FoundryRunTool         *self,
                                            GSubprocess            *subprocess);
DexFuture *foundry_run_tool_prepare        (FoundryRunTool         *self,
                                            FoundryBuildPipeline   *pipeline,
                                            FoundryCommand         *command,
                                            FoundryProcessLauncher *launcher,
                                            int                     pty_fd) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
