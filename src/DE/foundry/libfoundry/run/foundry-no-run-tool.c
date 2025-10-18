/* foundry-no-run-tool.c
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

#include "config.h"

#include <glib/gstdio.h>

#include "foundry-build-pipeline.h"
#include "foundry-command.h"
#include "foundry-debug.h"
#include "foundry-no-run-tool-private.h"
#include "foundry-process-launcher.h"
#include "foundry-sdk.h"

struct _FoundryNoRunTool
{
  FoundryRunTool parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryNoRunTool, foundry_no_run_tool, FOUNDRY_TYPE_RUN_TOOL)

typedef struct _Prepare
{
  FoundryBuildPipeline   *pipeline;
  FoundryCommand         *command;
  FoundryProcessLauncher *launcher;
  int                     pty_fd;
} Prepare;

static void
prepare_free (Prepare *state)
{
  g_clear_object (&state->pipeline);
  g_clear_object (&state->command);
  g_clear_object (&state->launcher);
  g_clear_fd (&state->pty_fd, NULL);
  g_free (state);
}

static DexFuture *
foundry_no_run_tool_prepare_fiber (gpointer data)
{
  Prepare *state = data;
  g_autoptr(GError) error = NULL;
  g_autofree char *cwd = NULL;
  g_auto(GStrv) argv = NULL;
  g_auto(GStrv) environ = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (state->pipeline));
  g_assert (FOUNDRY_IS_COMMAND (state->command));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (state->launcher));

  if ((argv = foundry_command_dup_argv (state->command)))
    foundry_process_launcher_append_args (state->launcher, (const char * const *)argv);

  if ((environ = foundry_command_dup_environ (state->command)))
    foundry_process_launcher_add_environ (state->launcher, (const char * const *)environ);

  if ((cwd = foundry_command_dup_cwd (state->command)))
    foundry_process_launcher_set_cwd (state->launcher, cwd);

  if (state->pty_fd > -1)
    {
      foundry_process_launcher_take_fd (state->launcher, dup (state->pty_fd), STDIN_FILENO);
      foundry_process_launcher_take_fd (state->launcher, dup (state->pty_fd), STDOUT_FILENO);
      foundry_process_launcher_take_fd (state->launcher, dup (state->pty_fd), STDERR_FILENO);
    }

  return dex_future_new_true ();
}

static DexFuture *
foundry_no_run_tool_prepare (FoundryRunTool         *run_tool,
                             FoundryBuildPipeline   *pipeline,
                             FoundryCommand         *command,
                             FoundryProcessLauncher *launcher,
                             int                     pty_fd)
{
  Prepare *state;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_NO_RUN_TOOL (run_tool));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_COMMAND (command));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  state = g_new0 (Prepare, 1);
  g_set_object (&state->pipeline, pipeline);
  g_set_object (&state->command, command);
  g_set_object (&state->launcher, launcher);
  state->pty_fd = dup (pty_fd);

  return dex_scheduler_spawn (NULL, 0,
                              foundry_no_run_tool_prepare_fiber,
                              state,
                              (GDestroyNotify) prepare_free);
}

static void
foundry_no_run_tool_class_init (FoundryNoRunToolClass *klass)
{
  FoundryRunToolClass *run_tool_class = FOUNDRY_RUN_TOOL_CLASS (klass);

  run_tool_class->prepare = foundry_no_run_tool_prepare;
}

static void
foundry_no_run_tool_init (FoundryNoRunTool *self)
{
}
