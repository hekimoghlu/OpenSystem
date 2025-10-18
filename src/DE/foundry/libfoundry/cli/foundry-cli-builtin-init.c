/* foundry-cli-builtin-init.c
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
#include "foundry-command-line.h"
#include "foundry-context.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_init_run (FoundryCommandLine *command_line,
                              const char * const *argv,
                              FoundryCliOptions  *options,
                              DexCancellable     *cancellable)
{
  FoundryContextFlags flags = FOUNDRY_CONTEXT_FLAGS_CREATE;
  g_autofree char *existing = NULL;
  g_autofree char *foundry_dir = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;
  const char *directory;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(directory = foundry_cli_options_get_string (options, "directory")))
    directory = foundry_command_line_get_directory (command_line);

  if ((existing = dex_await_string (foundry_context_discover (directory, cancellable), NULL)))
    {
      foundry_command_line_printerr (command_line,
                                     _("'%s' is already within a foundry project at '%s'"),
                                     directory, existing);
      foundry_command_line_printerr (command_line, "\n");
      return EXIT_FAILURE;
    }

  foundry_dir = g_build_filename (directory, ".foundry", NULL);

  if (!(context = dex_await_object (foundry_context_new (foundry_dir, directory, flags, cancellable), &error)))
    {
      foundry_command_line_printerr (command_line,
                                     "%s: %s: %s\n",
                                     _("error"),
                                     _("Failed to initialize foundry"),
                                     error->message);
      return EXIT_FAILURE;
    }

  foundry_command_line_print (command_line,
                              _("Initialized empty Foundry project at %s\n"),
                              foundry_dir);

  if (!dex_await (foundry_context_shutdown (context), &error))
    {
      foundry_command_line_printerr (command_line,
                                     "%s: %s: %s\n",
                                     _("error"),
                                     _("Failed to shutdown foundry"),
                                     error->message);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

void
foundry_cli_builtin_init (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "init"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "directory", 'd', 0, G_OPTION_ARG_FILENAME, NULL, N_("The directory to initialize, default is current"), N_("DIR") },
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_init_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("Initialize a foundry project"),
                                     });
}
