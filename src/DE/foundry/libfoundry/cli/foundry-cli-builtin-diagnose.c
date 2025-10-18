/* foundry-cli-builtin-diagnose.c
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
#include "foundry-context.h"
#include "foundry-diagnostic.h"
#include "foundry-diagnostic-manager.h"
#include "foundry-file-manager.h"
#include "foundry-model-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static char **
foundry_cli_builtin_diagnose_complete (FoundryCommandLine *command_line,
                                       const char         *command,
                                       const GOptionEntry *entry,
                                       FoundryCliOptions  *options,
                                       const char * const *argv,
                                       const char         *current)
{
  return g_strdupv ((char **)FOUNDRY_STRV_INIT ("__FOUNDRY_FILE"));
}

static int
foundry_cli_builtin_diagnose_run (FoundryCommandLine *command_line,
                                  const char * const *argv,
                                  FoundryCliOptions  *options,
                                  DexCancellable     *cancellable)
{
  FoundryObjectSerializerFormat format;
  g_autoptr(FoundryDiagnosticManager) diagnostic_manager = NULL;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListModel) list = NULL;
  g_autoptr(GPtrArray) files = NULL;
  g_autoptr(GError) error = NULL;
  const char *format_arg;

  static const FoundryObjectSerializerEntry fields[] = {
    { "path", N_("Path") },
    { "line", N_("Line") },
    { "line-offset", N_("Line Offset") },
    { "severity", N_("Severity") },
    { "message", N_("Message") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (argv[1] == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: %s FILE [FILE...]\n", argv[0]);
      return EXIT_FAILURE;
    }

  format_arg = foundry_cli_options_get_string (options, "format");
  format = foundry_object_serializer_format_parse (format_arg);

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  diagnostic_manager = foundry_context_dup_diagnostic_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (diagnostic_manager)), &error))
    goto handle_error;

  file_manager = foundry_context_dup_file_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (file_manager)), &error))
    goto handle_error;

  files = g_ptr_array_new_with_free_func (g_object_unref);

  for (guint i = 1; argv[i]; i++)
    g_ptr_array_add (files,
                     g_file_new_for_commandline_arg_and_cwd (argv[i],
                                                             foundry_command_line_get_directory (command_line)));

  if (!(list = dex_await_object (foundry_diagnostic_manager_diagnose_files (diagnostic_manager,
                                                                            (GFile **)files->pdata,
                                                                            files->len),
                                 &error)))
    goto handle_error;

  if (!dex_await (foundry_list_model_await (list), &error))
    goto handle_error;

  if (g_list_model_get_n_items (list) > 0)
    foundry_command_line_print_list (command_line, list, fields, format, FOUNDRY_TYPE_DIAGNOSTIC);

  return EXIT_SUCCESS;

handle_error:
  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_diagnose (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "diagnose"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_diagnose_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_diagnose_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("FILE [FILE...] - Diagnose a file"),
                                     });
}
