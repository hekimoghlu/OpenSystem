/* foundry-cli-builtin-lsp-prefer.c
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
#include "foundry-file-manager.h"
#include "foundry-lsp-manager.h"
#include "foundry-lsp-server.h"
#include "foundry-service.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_lsp_prefer_run (FoundryCommandLine *command_line,
                                    const char * const *argv,
                                    FoundryCliOptions  *options,
                                    DexCancellable     *cancellable)
{
  g_autoptr(FoundryLspManager) lsp_manager = NULL;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSettings) layer = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *language = NULL;
  g_auto(GStrv) languages = NULL;
  const char *module_name;
  gboolean project = FALSE;
  int argc;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  argc = g_strv_length ((char **)argv);

  if (argc < 3)
    {
      foundry_command_line_printerr (command_line, "usage: %s PLUGIN LANGUAGE\n", argv[0]);
      return EXIT_FAILURE;
    }

  module_name = argv[1];
  language = g_strdup (argv[2]);

  if (g_strcmp0 (module_name, "reset") == 0)
    {
      module_name = "";
    }
  else if (!peas_engine_get_plugin_info (peas_engine_get_default (), module_name))
    {
      foundry_command_line_printerr (command_line, "No plugin named \"%s\"\n", module_name);
      return EXIT_FAILURE;
    }

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  file_manager = foundry_context_dup_file_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (file_manager)), &error))
    goto handle_error;

  languages = foundry_file_manager_list_languages (file_manager);

  if (!g_strv_contains ((const char * const *)languages, language))
    {
      foundry_command_line_printerr (command_line, "No language named \"%s\"\n", language);
      return EXIT_FAILURE;
    }

  lsp_manager = foundry_context_dup_lsp_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (lsp_manager)), &error))
    goto handle_error;

  settings = foundry_lsp_manager_load_language_settings (lsp_manager, language);

  if (foundry_cli_options_get_boolean (options, "project", &project) && project)
    layer = foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_PROJECT);
  else
    layer = foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_USER);

  g_settings_set_string (layer, "preferred-module-name", module_name);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_lsp_prefer (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "lsp", "prefer"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "project", 'p', 0, G_OPTION_ARG_NONE, NULL, "Apply preference to project settings" },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_lsp_prefer_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("PLUGIN LANGUAGE"),
                                     });
}
