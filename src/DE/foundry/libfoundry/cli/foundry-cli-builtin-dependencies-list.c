/* foundry-cli-builtin-dependency-list.c
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
#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-context.h"
#include "foundry-dependency-manager.h"
#include "foundry-dependency.h"
#include "foundry-model-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_dependencies_list_run (FoundryCommandLine *command_line,
                                           const char * const *argv,
                                           FoundryCliOptions  *options,
                                           DexCancellable     *cancellable)
{
  FoundryObjectSerializerFormat format;
  g_autoptr(FoundryDependencyManager) dependency_manager = NULL;
  g_autoptr(FoundryConfigManager) config_manager = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GListModel) model = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;
  const char *format_arg;

  static const FoundryObjectSerializerEntry fields[] = {
    { "name", N_("Name") },
    { "kind", N_("Kind") },
    { "location", N_("Location") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  dependency_manager = foundry_context_dup_dependency_manager (foundry);
  config_manager = foundry_context_dup_config_manager (foundry);

  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (config_manager)), &error))
    goto handle_error;

  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (dependency_manager)), &error))
    goto handle_error;

  if (!(config = foundry_config_manager_dup_config (config_manager)))
    {
      foundry_command_line_printerr (command_line, "No active configuration\n");
      return EXIT_FAILURE;
    }

  if (!(model = dex_await_object (foundry_dependency_manager_list_dependencies (dependency_manager, config), &error)))
    goto handle_error;

  /* Wait for async population of model */
  dex_await (foundry_list_model_await (model), NULL);

  format_arg = foundry_cli_options_get_string (options, "format");
  format = foundry_object_serializer_format_parse (format_arg);
  foundry_command_line_print_list (command_line, G_LIST_MODEL (model), fields, format, FOUNDRY_TYPE_DEPENDENCY);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_dependencies_list (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "dependencies", "list"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_dependencies_list_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("List available dependencies"),
                                     });
}
