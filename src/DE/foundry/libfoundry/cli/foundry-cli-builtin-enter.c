/* foundry-cli-builtin-enter.c
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
#include <glib/gstdio.h>

#include "foundry-cli-builtin-private.h"
#include "foundry-command-line-remote-private.h"
#include "foundry-context.h"
#include "foundry-dbus-service.h"
#include "foundry-process-launcher.h"
#include "foundry-shell.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_enter_run (FoundryCommandLine *command_line,
                               const char * const *argv,
                               FoundryCliOptions  *options,
                               DexCancellable     *cancellable)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryDBusService) dbus_service = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *address = NULL;
  g_auto(GStrv) environ = NULL;
  g_autofree char *ident = NULL;

  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    {
      foundry_command_line_printerr (command_line, "%s: %s\n",
                                     _("error"),
                                     error->message);
      return EXIT_FAILURE;
    }

  dbus_service = foundry_context_dup_dbus_service (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (dbus_service)), &error))
    {
      foundry_command_line_printerr (command_line,
                                     "%s: %s\n",
                                     _("error"),
                                     error->message);
      return EXIT_FAILURE;
    }

  if (!(address = dex_await_string (foundry_dbus_service_query_address (dbus_service), &error)))
    {
      foundry_command_line_printerr (command_line, "%s: %s: %s\n",
                                     _("error"),
                                     _("Failed to setup D-Bus service"),
                                     error->message);
      return EXIT_FAILURE;
    }

  launcher = foundry_process_launcher_new ();

  /* Takeover the TTY from the current foreground (this process) */
  foundry_process_launcher_take_fd (launcher, dup (foundry_command_line_get_stdin (command_line)), STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher, dup (foundry_command_line_get_stdout (command_line)), STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher, dup (foundry_command_line_get_stderr (command_line)), STDERR_FILENO);

  /* Setup environment for the child shell */
  environ = foundry_command_line_get_environ (command_line);
  foundry_process_launcher_set_environ (launcher, (const char * const *)environ);
  foundry_process_launcher_setenv (launcher, "FOUNDRY_VERSION", PACKAGE_VERSION);
  foundry_process_launcher_setenv (launcher, "FOUNDRY_ADDRESS", address);

  foundry_process_launcher_set_cwd (launcher, foundry_command_line_get_directory (command_line));

  foundry_process_launcher_append_argv (launcher, foundry_shell_get_default ());

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    {
      foundry_command_line_printerr (command_line, "%s: %s: %s\n",
                                     _("error"),
                                     _("Failed to spawn shell"),
                                     error->message);
      return EXIT_FAILURE;
    }

  ident = g_strdup (g_subprocess_get_identifier (subprocess));

  if (!dex_await (dex_subprocess_wait_check (subprocess), &error))
    {
      if (g_subprocess_get_if_signaled (subprocess))
        foundry_command_line_printerr (command_line, "%s: %s: %s\n",
                                       _("error"),
                                       _("Child shell exited"),
                                       error->message);
      return g_subprocess_get_exit_status (subprocess);
    }

  g_debug ("Child process \"%s\" exited cleanly", ident);

  return EXIT_SUCCESS;
}


void
foundry_cli_builtin_enter (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "enter"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_enter_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("Start a shell within a foundry project"),
                                     });
}
