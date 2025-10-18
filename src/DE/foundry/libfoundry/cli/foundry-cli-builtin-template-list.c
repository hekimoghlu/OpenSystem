/* foundry-cli-builtin-template-list.c
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
#include "foundry-init-private.h"
#include "foundry-model-manager.h"
#include "foundry-project-template.h"
#include "foundry-template-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_template_list_run (FoundryCommandLine *command_line,
                                       const char * const *argv,
                                       FoundryCliOptions  *options,
                                       DexCancellable     *cancellable)
{
  FoundryObjectSerializerFormat format;
  g_autoptr(FoundryTemplateManager) template_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListModel) templates = NULL;
  g_autoptr(DexFuture) future = NULL;
  g_autoptr(GError) error = NULL;
  const char *format_arg;

  static const FoundryObjectSerializerEntry fields[] = {
    { "id", N_("ID") },
    { "description", N_("Description") },
    { "tags", N_("Tags") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  /* Optional context, okay to continue without */
  context = dex_await_object (foundry_cli_options_load_context (options, command_line), NULL);

  template_manager = foundry_template_manager_new ();

  if (!(templates = dex_await_object (foundry_template_manager_list_templates (template_manager, context), &error)))
    goto handle_error;

  dex_await (foundry_list_model_await (templates), NULL);

  format_arg = foundry_cli_options_get_string (options, "format");
  format = foundry_object_serializer_format_parse (format_arg);
  foundry_command_line_print_list (command_line, templates, fields, format, FOUNDRY_TYPE_TEMPLATE);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_template_list (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "template", "list"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_template_list_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("List available project templates"),
                                     });
}
