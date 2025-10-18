/* foundry-cli-builtin-run.c
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

#include "config.h"

#include <glib/gi18n-lib.h>

#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command.h"
#include "foundry-command-line.h"
#include "foundry-config.h"
#include "foundry-context.h"
#include "foundry-run-manager.h"
#include "foundry-run-tool.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_run_run (FoundryCommandLine *command_line,
                             const char * const *argv,
                             FoundryCliOptions  *options,
                             DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryRunManager) run_manager = NULL;
  g_autoptr(FoundryCommand) default_command = NULL;
  g_autoptr(FoundryRunTool) tool = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  run_manager = foundry_context_dup_run_manager (context);
  build_manager = foundry_context_dup_build_manager (context);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    goto handle_error;

  config = foundry_build_pipeline_dup_config (pipeline);

  if (argv[1] && g_str_equal (argv[1], "--") && argv[2])
    {
      g_autofree char *cwd = foundry_command_line_get_directory (command_line);

      /* Do not pass environ which might mess up how things run. Require the
       * user to use 'env ...' in that case.
       */

      default_command = foundry_command_new (context);
      foundry_command_set_argv (default_command, &argv[2]);
      foundry_command_set_cwd (default_command, cwd);
    }

  if (default_command == NULL)
    default_command = foundry_config_dup_default_command (config);

  if (default_command == NULL)
    {
      foundry_command_line_printerr (command_line,
                                     "%s\n",
                                     _("No default run command specified in configuration"));
      return EXIT_FAILURE;
    }

  if (!(tool = dex_await_object (foundry_run_manager_run_command (run_manager,
                                                                  pipeline,
                                                                  default_command,
                                                                  NULL,
                                                                  foundry_command_line_get_stdout (command_line),
                                                                  foundry_command_line_get_stdout (command_line),
                                                                  cancellable),
                                 &error)))
    goto handle_error;

  if (!dex_await (dex_future_first (dex_ref (cancellable),
                                    foundry_run_tool_await (tool),
                                    NULL),
                  &error))
    {
      dex_await (foundry_run_tool_force_exit (tool), NULL);
      goto handle_error;
    }

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_run (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "run"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_run_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("[-- COMMAND...] - Run the application"),
                                     });
}
