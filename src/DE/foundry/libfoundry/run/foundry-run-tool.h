/* foundry-run-tool.h
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

#include <libpeas.h>

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_RUN_TOOL (foundry_run_tool_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryRunTool, foundry_run_tool, FOUNDRY, RUN_TOOL, FoundryContextual)

struct _FoundryRunToolClass
{
  FoundryContextualClass parent_class;

  void       (*started)     (FoundryRunTool         *self,
                             GSubprocess            *subprocess);
  void       (*stopped)     (FoundryRunTool         *self);
  DexFuture *(*prepare)     (FoundryRunTool         *self,
                             FoundryBuildPipeline   *pipeline,
                             FoundryCommand         *command,
                             FoundryProcessLauncher *launcher,
                             int                     pty_fd);
  DexFuture *(*force_exit)  (FoundryRunTool         *self);
  DexFuture *(*send_signal) (FoundryRunTool         *self,
                             int                     signum);
};

FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo *foundry_run_tool_dup_plugin_info (FoundryRunTool         *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_run_tool_force_exit      (FoundryRunTool         *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_run_tool_send_signal     (FoundryRunTool         *self,
                                                  int                     signum) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_run_tool_await           (FoundryRunTool         *self) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
