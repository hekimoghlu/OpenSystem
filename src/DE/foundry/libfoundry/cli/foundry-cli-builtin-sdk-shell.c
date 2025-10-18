/* foundry-cli-builtin-sdk-shell.c
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

#include "foundry-cli-builtin-private.h"
#include "foundry-context.h"
#include "foundry-sdk.h"
#include "foundry-sdk-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static char **
foundry_cli_builtin_sdk_shell_complete (FoundryCommandLine *command_line,
                                        const char         *command,
                                        const GOptionEntry *entry,
                                        FoundryCliOptions  *options,
                                        const char * const *argv,
                                        const char         *current)
{
  return foundry_cli_builtin_complete_list_model (options, command_line,
                                                  argv, current,
                                                  "sdk-manager", "id");
}

static void
foundry_cli_builtin_sdk_shell_help (FoundryCommandLine *command_line)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  foundry_command_line_print (command_line, "Usage:\n");
  foundry_command_line_print (command_line, "  foundry sdk shell [OPTIONS…] SDK\n");
  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Options:\n");
  foundry_command_line_print (command_line, "  --help                Show help options\n");
  foundry_command_line_print (command_line, "\n");
}

static int
foundry_cli_builtin_sdk_shell_run (FoundryCommandLine *command_line,
                                    const char * const *argv,
                                    FoundryCliOptions  *options,
                                    DexCancellable     *cancellable)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *sdk_id = NULL;
  g_autofree char *path = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (foundry_cli_options_help (options))
    {
      foundry_cli_builtin_sdk_shell_help (command_line);
      return EXIT_SUCCESS;
    }

  if (argv[1] == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: foundry sdk shell SDK\n");
      return EXIT_FAILURE;
    }

  sdk_id = g_strdup (argv[1]);

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  sdk_manager = foundry_context_dup_sdk_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (sdk_manager)), &error))
    goto handle_error;

  if (!(sdk = dex_await_object (foundry_sdk_manager_find_by_id (sdk_manager, sdk_id), NULL)))
    {
      foundry_command_line_printerr (command_line, "No such SDK \"%s\"\n", sdk_id);
      return EXIT_FAILURE;
    }

  if (!(path = dex_await_string (foundry_sdk_discover_shell (sdk), &error)))
    {
      foundry_command_line_printerr (command_line,
                                     "Failed to locate shell: %s\n",
                                     error->message);
      return EXIT_FAILURE;
    }

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_sdk_prepare_to_build (sdk, NULL, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
    {
      foundry_command_line_printerr (command_line,
                                     "Failed to prepare SDK: %s\n",
                                     error->message);
      return EXIT_FAILURE;
    }

  foundry_process_launcher_append_argv (launcher, path);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stdin (command_line)),
                                    STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stdout (command_line)),
                                    STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher,
                                    dup (foundry_command_line_get_stderr (command_line)),
                                    STDERR_FILENO);

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
foundry_cli_builtin_sdk_shell (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "sdk", "shell"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_sdk_shell_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_sdk_shell_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("SDK - Start a shell in SDK"),
                                     });
}
