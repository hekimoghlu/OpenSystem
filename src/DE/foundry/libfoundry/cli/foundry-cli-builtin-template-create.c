/* foundry-cli-builtin-template-create.c
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
#include "foundry-input.h"
#include "foundry-internal-template-private.h"
#include "foundry-model-manager.h"
#include "foundry-project-template.h"
#include "foundry-template-manager.h"
#include "foundry-template-output.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static char **
foundry_cli_builtin_template_create_complete (FoundryCommandLine *command_line,
                                              const char         *command,
                                              const GOptionEntry *entry,
                                              FoundryCliOptions  *options,
                                              const char * const *argv,
                                              const char         *current)
{
  g_autoptr(FoundryTemplateManager) template_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GStrvBuilder) builder = NULL;
  g_autoptr(GListModel) model = NULL;

  context = dex_await_object (foundry_cli_options_load_context (options, command_line), NULL);
  template_manager = foundry_template_manager_new ();

  if (!(model = dex_await_object (foundry_template_manager_list_templates (template_manager, context), NULL)))
    return NULL;

  dex_await (foundry_list_model_await (model), NULL);

  return foundry_cli_builtin_complete_model (model, argv, current, "id");
}

static int
foundry_cli_builtin_template_create_run (FoundryCommandLine *command_line,
                                         const char * const *argv,
                                         FoundryCliOptions  *options,
                                         DexCancellable     *cancellable)
{
  g_autoptr(FoundryTemplate) template = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryInput) input = NULL;
  g_autoptr(GListModel) outputs = NULL;
  g_autoptr(GFile) directory = NULL;
  g_autoptr(GFile) template_file = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *directory_path = NULL;
  const char *template_id;
  guint n_items;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (g_strv_length ((char **)argv) != 2)
    {
      foundry_command_line_printerr (command_line, "usage: %s TEMPLATE_ID|TEMPLATE_FILE\n", argv[0]);
      return EXIT_FAILURE;
    }

  context = dex_await_object (foundry_cli_options_load_context (options, command_line), NULL);

  template_id = argv[1];
  template_file = foundry_command_line_build_file_for_arg (command_line, template_id);

  if (dex_await_boolean (dex_file_query_exists (template_file), NULL))
    {
      if (!(template = dex_await_object (foundry_internal_template_new (context, template_file), &error)))
        goto handle_error;
    }
  else
    {
      g_autoptr(FoundryTemplateManager) template_manager = NULL;

      template_manager = foundry_template_manager_new ();

      if (!(template = dex_await_object (foundry_template_manager_find_template (template_manager, context, template_id), &error)))
        goto handle_error;
    }

  if ((input = foundry_template_dup_input (template)))
    {
      if (!dex_await (foundry_command_line_request_input (command_line, input), &error))
        goto handle_error;
    }

  if (!(outputs = dex_await_object (foundry_template_expand (template), &error)))
    goto handle_error;

  directory_path = foundry_command_line_get_directory (command_line);
  directory = g_file_new_for_path (directory_path);
  n_items = g_list_model_get_n_items (outputs);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTemplateOutput) output = g_list_model_get_item (outputs, i);
      g_autoptr(GFile) file = foundry_template_output_dup_file (output);
      g_autofree char *path = NULL;

      if (g_file_has_prefix (file, directory))
        path = g_file_get_relative_path (directory, file);
      else
        path = g_file_get_path (file);

      foundry_command_line_print (command_line, "%s\n", path);

      if (!dex_await (foundry_template_output_write (output), &error))
        goto handle_error;
    }

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_template_create (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "template", "create"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_template_create_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_template_create_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("TEMPLATE_ID|TEMPLATE_FILE - Expand a template"),
                                     });
}
