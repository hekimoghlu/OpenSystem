/* foundry-cli-builtin-vcs-list-files.c
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
#include "foundry-cli-command-tree.h"
#include "foundry-command-line.h"
#include "foundry-context.h"
#include "foundry-vcs.h"
#include "foundry-vcs-file.h"
#include "foundry-vcs-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_vcs_list_files_run (FoundryCommandLine *command_line,
                                        const char * const *argv,
                                        FoundryCliOptions  *options,
                                        DexCancellable     *cancellable)
{
  FoundryObjectSerializerFormat format;
  g_autoptr(FoundryVcsManager) vcs_manager = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(FoundryVcs) vcs = NULL;
  g_autoptr(GListModel) list = NULL;
  g_autoptr(GError) error = NULL;
  const char *format_arg;

  static const FoundryObjectSerializerEntry fields[] = {
    { "relative-path", N_("Path") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  vcs_manager = foundry_context_dup_vcs_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (vcs_manager)), &error))
    goto handle_error;

  vcs = foundry_vcs_manager_dup_vcs (vcs_manager);
  if (!(list = dex_await_object (foundry_vcs_list_files (vcs), &error)))
    goto handle_error;

  format_arg = foundry_cli_options_get_string (options, "format");
  format = foundry_object_serializer_format_parse (format_arg);

  if (format == FOUNDRY_OBJECT_SERIALIZER_FORMAT_TEXT)
    {
      guint n_items = g_list_model_get_n_items (list);

      /* Just use g_print to avoid space calculations/headers */

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryVcsFile) file = g_list_model_get_item (list, i);
          g_autofree char *path = foundry_vcs_file_dup_relative_path (file);

          foundry_command_line_print (command_line, "%s\n", path);
        }
    }
  else
    {
      foundry_command_line_print_list (command_line, list, fields, format, G_TYPE_INVALID);
    }

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_vcs_list_files (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "vcs", "list-files"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_vcs_list_files_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("List files known to the VCS"),
                                     });
}
