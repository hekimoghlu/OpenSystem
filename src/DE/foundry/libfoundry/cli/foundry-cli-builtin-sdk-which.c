/* foundry-cli-builtin-sdk-which.c
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
foundry_cli_builtin_sdk_which_complete (FoundryCommandLine *command_line,
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
foundry_cli_builtin_sdk_which_help (FoundryCommandLine *command_line)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  foundry_command_line_print (command_line, "Usage:\n");
  foundry_command_line_print (command_line, "  foundry sdk which [OPTIONS…] SDK PROGRAM\n");
  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Options:\n");
  foundry_command_line_print (command_line, "  --help                Show help options\n");
  foundry_command_line_print (command_line, "\n");
}

static int
foundry_cli_builtin_sdk_which_run (FoundryCommandLine *command_line,
                                   const char * const *argv,
                                   FoundryCliOptions  *options,
                                   DexCancellable     *cancellable)
{
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *sdk_id = NULL;
  g_autofree char *path = NULL;
  g_autofree char *program = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (foundry_cli_options_help (options))
    {
      foundry_cli_builtin_sdk_which_help (command_line);
      return EXIT_SUCCESS;
    }

  if (argv[1] == NULL || argv[2] == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: foundry sdk which SDK PROGRAM\n");
      return EXIT_FAILURE;
    }

  sdk_id = g_strdup (argv[1]);
  program = g_strdup (argv[2]);

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

  if (!foundry_sdk_get_installed (sdk))
    {
      foundry_command_line_printerr (command_line, "SDK \"%s\" is not installed.\n", sdk_id);
      return EXIT_FAILURE;
    }

  if (!(path = dex_await_string (foundry_sdk_contains_program (sdk, program), &error)))
    {
      foundry_command_line_printerr (command_line, "No such command \"%s\" in SDK \"%s\"\n",
                                     program, sdk_id);
      return EXIT_FAILURE;
    }

  foundry_command_line_printerr (command_line, "%s\n", path);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_sdk_which (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "sdk", "which"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_sdk_which_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_sdk_which_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("SDK PROGRAM - Look for PROGRAM in SDK"),
                                     });
}
