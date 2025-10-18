/* foundry-cli-builtin-reveal.c
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

#include "foundry-file-manager.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-private.h"
#include "foundry-context.h"
#include "foundry-util-private.h"

static void
foundry_cli_builtin_show_help (FoundryCommandLine *command_line)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  foundry_command_line_print (command_line, "Usage:\n");
  foundry_command_line_print (command_line, "  foundry show FILE\n");
  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Options:\n");
  foundry_command_line_print (command_line, "  -h, --help   Show help options\n");
  foundry_command_line_print (command_line, "\n");
}

static char **
foundry_cli_builtin_show_complete (FoundryCommandLine *command_line,
                                   const char         *command,
                                   const GOptionEntry *entry,
                                   FoundryCliOptions  *options,
                                   const char * const *argv,
                                   const char         *current)
{
  return g_strdupv ((char **)FOUNDRY_STRV_INIT ("__FOUNDRY_FILE"));
}

static int
foundry_cli_builtin_show_run (FoundryCommandLine *command_line,
                              const char * const *argv,
                              FoundryCliOptions  *options,
                              DexCancellable     *cancellable)
{
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GFile) file = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (argv[1] == NULL || foundry_cli_options_help (options))
    {
      foundry_cli_builtin_show_help (command_line);
      return EXIT_SUCCESS;
    }

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  file_manager = foundry_context_dup_file_manager (foundry);
  file = g_file_new_for_commandline_arg_and_cwd (argv[1],
                                                 foundry_command_line_get_directory (command_line));

  if (!dex_await (foundry_file_manager_show (file_manager, file), &error))
    goto handle_error;

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_show (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "show"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_show_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_show_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("Reveal a file in the file manager"),
                                     });
}
