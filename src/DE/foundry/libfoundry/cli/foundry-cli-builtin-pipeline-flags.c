/* foundry-cli-builtin-pipeline-flags.c
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

#include "foundry-build-flags.h"
#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-build-progress.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-private.h"
#include "foundry-context.h"
#include "foundry-util-private.h"

static char **
foundry_cli_builtin_pipeline_flags_complete (FoundryCommandLine *command_line,
                                             const char         *command,
                                             const GOptionEntry *entry,
                                             FoundryCliOptions  *options,
                                             const char * const *argv,
                                             const char         *current)
{
  return g_strdupv ((char **)FOUNDRY_STRV_INIT ("__FOUNDRY_FILE"));
}

static int
foundry_cli_builtin_pipeline_flags_run (FoundryCommandLine *command_line,
                                        const char * const *argv,
                                        FoundryCliOptions  *options,
                                        DexCancellable     *cancellable)
{
  FoundryObjectSerializerFormat format;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildFlags) build_flags = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) file = NULL;
  g_autofree char *existing = NULL;
  const char *format_arg;

  static const FoundryObjectSerializerEntry fields[] = {
    { "directory", N_("Directory") },
    { "flags", N_("Flags") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (argv[1] == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: %s FILE\n", argv[0]);
      return EXIT_FAILURE;
    }

  file = g_file_new_build_filename (foundry_command_line_get_directory (command_line), argv[1], NULL);

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  build_manager = foundry_context_dup_build_manager (context);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    goto handle_error;

  if (!(build_flags = dex_await_object (foundry_build_pipeline_find_build_flags (pipeline, file), &error)))
    goto handle_error;

  format_arg = foundry_cli_options_get_string (options, "format");
  format = foundry_object_serializer_format_parse (format_arg);
  foundry_command_line_print_object (command_line, G_OBJECT (build_flags), fields, format);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "Error: %s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_pipeline_flags (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "pipeline", "flags"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_pipeline_flags_run,
                                       .complete = foundry_cli_builtin_pipeline_flags_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("FILE - List command to compile file"),
                                     });
}
