/* foundry-cli-builtin-guess-language.c
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

#include <glib/gi18n-lib.h>

#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command-line.h"
#include "foundry-config.h"
#include "foundry-context.h"
#include "foundry-file-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static char **
foundry_cli_builtin_guess_language_complete (FoundryCommandLine *command_line,
                                             const char         *command,
                                             const GOptionEntry *entry,
                                             FoundryCliOptions  *options,
                                             const char * const *argv,
                                             const char         *current)
{
  return g_strdupv ((char **)FOUNDRY_STRV_INIT ("__FOUNDRY_FILE"));
}

static int
foundry_cli_builtin_guess_language_run (FoundryCommandLine *command_line,
                                        const char * const *argv,
                                        FoundryCliOptions  *options,
                                        DexCancellable     *cancellable)
{
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GFileInfo) info = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) file = NULL;
  g_autofree char *language = NULL;
  const char *content_type = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (argv[1] == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: %s FILE\n", argv[0]);
      return EXIT_FAILURE;
    }

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  file_manager = foundry_context_dup_file_manager (context);
  file = g_file_new_for_commandline_arg_and_cwd (argv[1], foundry_command_line_get_directory (command_line));

  /* Try to get conetent-type for file first */
  info = dex_await_object (dex_file_query_info (file,
                                                G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
                                                G_FILE_QUERY_INFO_NONE,
                                                G_PRIORITY_DEFAULT),
                           NULL);

  if (info != NULL)
    content_type = g_file_info_get_content_type (info);

  if (!(language = dex_await_string (foundry_file_manager_guess_language (file_manager, file, content_type, NULL), &error)))
    goto handle_error;

  foundry_command_line_print (command_line, "%s\n", language);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_guess_language (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "guess-language"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_guess_language_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_guess_language_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("FILE - Guess a files language"),
                                     });
}
