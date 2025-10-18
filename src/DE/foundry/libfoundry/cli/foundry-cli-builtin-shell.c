/* foundry-cli-builtin-shell.c
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

#include "foundry-build-pipeline.h"
#include "foundry-build-manager.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-context.h"
#include "foundry-sdk.h"
#include "foundry-sdk-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_shell_run (FoundryCommandLine *command_line,
                               const char * const *argv,
                               FoundryCliOptions  *options,
                               DexCancellable     *cancellable)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *shell = NULL;
  g_autofree char *builddir = NULL;
  DexFuture *future;
  gboolean run = FALSE;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  foundry_cli_options_get_boolean (options, "run", &run);

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  sdk_manager = foundry_context_dup_sdk_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (sdk_manager)), &error))
    goto handle_error;

  if (!(sdk = foundry_sdk_manager_dup_sdk (sdk_manager)))
    {
      foundry_command_line_printerr (command_line,
                                     "SDK is not configured, cannot load environment\n");
      return EXIT_FAILURE;
    }

  if (!(shell = dex_await_string (foundry_sdk_discover_shell (sdk), &error)))
    {
      foundry_command_line_printerr (command_line,
                                     "Cannot discover shell: %s\n",
                                     error->message);
      return EXIT_FAILURE;
    }

  launcher = foundry_process_launcher_new ();

  build_manager = foundry_context_dup_build_manager (context);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    goto handle_error;

  builddir = foundry_build_pipeline_dup_builddir (pipeline);

  if (run)
    future = foundry_sdk_prepare_to_run (sdk, pipeline, launcher);
  else
    future = foundry_sdk_prepare_to_build (sdk, pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD);

  if (!dex_await (g_steal_pointer (&future), &error))
    {
      foundry_command_line_printerr (command_line,
                                     "Failed to prepare SDK: %s\n",
                                     error->message);
      return EXIT_FAILURE;
    }

  foundry_process_launcher_append_argv (launcher, shell);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stdin (command_line)),
                                    STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stdout (command_line)),
                                    STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stderr (command_line)),
                                    STDERR_FILENO);

  foundry_process_launcher_set_cwd (launcher, builddir);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    {
      foundry_command_line_printerr (command_line,
                                     "Failed to spawn shell: %s\n",
                                     error->message);
      return EXIT_FAILURE;
    }

  dex_await (dex_subprocess_wait_check (subprocess), NULL);

  if (g_subprocess_get_if_exited (subprocess))
    return g_subprocess_get_exit_status (subprocess);

  foundry_command_line_printerr (command_line,
                                 "Shell exited with signal %u\n",
                                 g_subprocess_get_term_sig (subprocess));

  return EXIT_FAILURE;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_shell (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "shell"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "run", 'r', 0, G_OPTION_ARG_NONE, NULL, _("Enter environment for running applications") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_shell_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("- Start shell in build environment"),
                                     });
}
