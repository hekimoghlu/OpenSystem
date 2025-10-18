/* foundry-cli-builtin-vcs-fetch.c
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

#include "line-reader-private.h"

#include "foundry-auth-provider.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command-line.h"
#include "foundry-context.h"
#include "foundry-operation.h"
#include "foundry-vcs.h"
#include "foundry-vcs-remote.h"
#include "foundry-vcs-manager.h"
#include "foundry-service.h"

static char **
foundry_cli_builtin_vcs_fetch_complete (FoundryCommandLine *command_line,
                                        const char         *command,
                                        const GOptionEntry *entry,
                                        FoundryCliOptions  *options,
                                        const char * const *argv,
                                        const char         *current)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryVcsManager) vcs_manager = NULL;
  g_autoptr(FoundryVcs) vcs = NULL;
  g_autoptr(GListModel) list = NULL;
  g_autoptr(GStrvBuilder) builder = NULL;

  if (g_strv_length ((char **)argv) > 2 ||
      (g_strv_length ((char **)argv) == 2 && foundry_str_empty0 (current)))
    return NULL;

  builder = g_strv_builder_new ();

  if ((context = dex_await_object (foundry_cli_options_load_context (options, command_line), NULL)) &&
      (vcs_manager = foundry_context_dup_vcs_manager (context)) &&
      (vcs = foundry_vcs_manager_dup_vcs (vcs_manager)) &&
      (list = dex_await_object (foundry_vcs_list_remotes (vcs), NULL)))
    {
      guint n_items = g_list_model_get_n_items (list);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryVcsRemote) remote = g_list_model_get_item (list, i);
          g_autofree char *name = foundry_vcs_remote_dup_name (remote);

          g_strv_builder_add (builder, name);
        }
    }

  return g_strv_builder_end (builder);
}

static int
foundry_cli_builtin_vcs_fetch_run (FoundryCommandLine *command_line,
                                   const char * const *argv,
                                   FoundryCliOptions  *options,
                                   DexCancellable     *cancellable)
{
  g_autoptr(FoundryAuthProvider) auth_provider = NULL;
  g_autoptr(FoundryVcsManager) vcs_manager = NULL;
  g_autoptr(FoundryVcsRemote) remote = NULL;
  g_autoptr(FoundryOperation) operation = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryVcs) vcs = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (g_strv_length ((char **)argv) < 2)
    {
      foundry_command_line_printerr (command_line, "usage: %s REMOTE\n", argv[0]);
      return EXIT_FAILURE;
    }

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  vcs_manager = foundry_context_dup_vcs_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (vcs_manager)), &error))
    goto handle_error;

  if (!(vcs = foundry_vcs_manager_dup_vcs (vcs_manager)))
    {
      foundry_command_line_printerr (command_line, "No VCS in use.\n");
      return EXIT_FAILURE;
    }

  if (!(remote = dex_await_object (foundry_vcs_find_remote (vcs, argv[1]), &error)))
    goto handle_error;

  operation = foundry_operation_new ();

  if (!(auth_provider = foundry_command_line_dup_auth_provider (command_line)))
    auth_provider = foundry_auth_provider_new_for_context (context);

  foundry_operation_set_auth_provider (operation, auth_provider);

  if (!(remote = dex_await_object (foundry_vcs_find_remote (vcs, argv[1]), &error)))
    goto handle_error;

  if (!dex_await (foundry_vcs_fetch (vcs, remote, operation), &error))
    goto handle_error;

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_vcs_fetch (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "vcs", "fetch"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_vcs_fetch_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_vcs_fetch_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("FILE - Print commit fetch information"),
                                     });
}
