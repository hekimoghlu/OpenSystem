/* foundry-cli-builtin-test-run.c
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
#include "foundry-command.h"
#include "foundry-context.h"
#include "foundry-model-manager.h"
#include "foundry-process-launcher.h"
#include "foundry-test-manager.h"
#include "foundry-test.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_test_run_run (FoundryCommandLine *command_line,
                                  const char * const *argv,
                                  FoundryCliOptions  *options,
                                  DexCancellable     *cancellable)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryTestManager) test_manager = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(FoundryTest) test = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  const char *test_name;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (g_strv_length ((char **)argv) < 2)
    {
      foundry_command_line_printerr (command_line, "usage: %s TEST_NAME\n", argv[0]);
      return EXIT_FAILURE;
    }

  test_name = argv[1];

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  test_manager = foundry_context_dup_test_manager (foundry);
  build_manager = foundry_context_dup_build_manager (foundry);

  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (test_manager)), &error) ||
      !dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (build_manager)), &error))
    goto handle_error;

  if (!(test = dex_await_object (foundry_test_manager_find_test (test_manager, test_name), &error)))
    goto handle_error;

  if (!(command = foundry_test_dup_command (test)))
    {
      foundry_command_line_printerr (command_line, "Error: `%s` is missing a test command", test_name);
      return EXIT_FAILURE;
    }

  pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), NULL);
  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_command_prepare (command, pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
    goto handle_error;

  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stdout (command_line)),
                                    STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stderr (command_line)),
                                    STDERR_FILENO);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    goto handle_error;

  if (!dex_await (dex_subprocess_wait_check (subprocess), &error))
    goto handle_error;

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_test_run (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test", "run"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_test_run_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("TEST - Run a test"),
                                     });
}
