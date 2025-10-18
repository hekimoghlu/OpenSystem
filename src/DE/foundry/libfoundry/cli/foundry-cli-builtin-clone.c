/* foundry-cli-builtin-clone.c
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
#include "foundry-command-line.h"
#include "foundry-git-cloner.h"
#include "foundry-git-uri.h"
#include "foundry-operation.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_clone_run (FoundryCommandLine *command_line,
                               const char * const *argv,
                               FoundryCliOptions  *options,
                               DexCancellable     *cancellable)
{
  g_autoptr(FoundryGitCloner) cloner = NULL;
  g_autoptr(FoundryOperation) operation = NULL;
  g_autoptr(FoundryGitUri) uri = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) file = NULL;
  g_autoptr(GFile) final_dir = NULL;
  const char *branch;
  const char *directory;
  gboolean bare = FALSE;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (g_strv_length ((char **)argv) != 2)
    {
      g_printerr ("usage: %s [OPTIONS] URI\n", argv[0]);
      return EXIT_FAILURE;
    }

  if (!(uri = foundry_git_uri_new (argv[1])))
    {
      g_printerr ("Invalid URI: `%s`\n", argv[1]);
      return EXIT_FAILURE;
    }

  cloner = foundry_git_cloner_new ();
  foundry_git_cloner_set_uri (cloner, argv[1]);

  if (!(directory = foundry_cli_options_get_filename (options, "directory")))
    {
      directory = foundry_command_line_get_directory (command_line);

      if (!g_path_is_absolute (directory))
        {
          g_printerr ("Expected absolute directory but got `%s`", directory);
          return EXIT_FAILURE;
        }

      file = g_file_new_for_path (directory);
      final_dir = g_file_get_child (file, foundry_git_uri_get_clone_name (uri));
    }
  else if (!g_path_is_absolute (directory))
    {
      file = g_file_new_build_filename (foundry_command_line_get_directory (command_line), directory, NULL);
      final_dir = g_object_ref (file);
    }

  foundry_git_cloner_set_directory (cloner, final_dir);

  if (foundry_cli_options_get_boolean (options, "bare", &bare))
    foundry_git_cloner_set_bare (cloner, bare);

  if ((branch = foundry_cli_options_get_string (options, "branch")))
    foundry_git_cloner_set_remote_branch_name(cloner, branch);

  operation = foundry_operation_new ();

  if (!dex_await (foundry_git_cloner_clone (cloner, foundry_command_line_get_stdout (command_line), operation), &error))
    {
      g_printerr ("%s\n", error->message);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

void
foundry_cli_builtin_clone (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "clone"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "directory", 'd', 0, G_OPTION_ARG_FILENAME, NULL, N_("The directory to initialize, default is current"), N_("DIR") },
                                         { "branch", 0, 0, G_OPTION_ARG_STRING, NULL, N_("Specify a branch name to clone"), N_("Branch") },
                                         { "bare", 0, 0, G_OPTION_ARG_NONE },
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_clone_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("URI - Clone a Git repository"),
                                     });
}
