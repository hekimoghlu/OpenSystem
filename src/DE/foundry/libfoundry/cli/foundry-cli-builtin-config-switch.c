/* foundry-cli-builtin-config-switch.c
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
#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-service.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

static char **
foundry_cli_builtin_config_switch_complete (FoundryCommandLine *command_line,
                                            const char         *command,
                                            const GOptionEntry *entry,
                                            FoundryCliOptions  *options,
                                            const char * const *argv,
                                            const char         *current)
{
  return foundry_cli_builtin_complete_list_model (options, command_line,
                                                  argv, current,
                                                  "config-manager", "id");
}

static void
foundry_cli_builtin_config_switch_help (FoundryCommandLine *command_line)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  foundry_command_line_print (command_line, "Usage:\n");
  foundry_command_line_print (command_line, "  foundry config switch [OPTIONS…] CONFIG_ID\n");
  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Options:\n");
  foundry_command_line_print (command_line, "  --help                Show help options\n");
  foundry_command_line_print (command_line, "\n");
}

static int
foundry_cli_builtin_config_switch_run (FoundryCommandLine *command_line,
                                       const char * const *argv,
                                       FoundryCliOptions  *options,
                                       DexCancellable     *cancellable)
{
  g_autoptr(FoundryConfigManager) config_manager = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *config_id = NULL;
  gboolean project = FALSE;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  foundry_cli_options_get_boolean (options, "project", &project);

  if (foundry_cli_options_help (options))
    {
      foundry_cli_builtin_config_switch_help (command_line);
      return EXIT_SUCCESS;
    }

  config_id = g_strdup (argv[1]);

  if (config_id == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: foundry config switch CONFIG_ID\n");
      return EXIT_FAILURE;
    }

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  config_manager = foundry_context_dup_config_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (config_manager)), &error))
    goto handle_error;

  if (!(config = foundry_config_manager_find_config (config_manager, config_id)))
    {
      foundry_command_line_printerr (command_line, "No such config \"%s\"\n", config_id);
      return EXIT_FAILURE;
    }

  foundry_config_manager_set_config (config_manager, config);

  if (project)
    {
      g_autoptr(FoundrySettings) settings = foundry_context_load_project_settings (foundry);
      g_autoptr(GSettings) gsettings = foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_PROJECT);

      g_settings_set_string (gsettings, "config", config_id);
    }

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_config_switch (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "config", "switch"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "project", 'p', 0, G_OPTION_ARG_NONE, NULL, N_("Set config as default for all project contributors") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_config_switch_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_config_switch_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("CONFIG - Switch current config"),
                                     });
}
