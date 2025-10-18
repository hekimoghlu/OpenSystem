/* foundry-cli-builtin-pipeline-invalidate.c
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

#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-build-progress.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-private.h"
#include "foundry-context.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_pipeline_invalidate_run (FoundryCommandLine *command_line,
                                             const char * const *argv,
                                             FoundryCliOptions  *options,
                                             DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    {
      foundry_command_line_printerr (command_line, "%s\n", error->message);
      return EXIT_FAILURE;
    }

  build_manager = foundry_context_dup_build_manager (context);

  if (dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (build_manager)), NULL))
    foundry_build_manager_invalidate (build_manager);

  return EXIT_SUCCESS;
}

void
foundry_cli_builtin_pipeline_invalidate (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "pipeline", "invalidate"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_pipeline_invalidate_run,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("Invalidate the build pipeline"),
                                     });
}
