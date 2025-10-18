/* foundry-cli-builtin-deploy.c
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

#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command.h"
#include "foundry-command-line.h"
#include "foundry-config.h"
#include "foundry-context.h"
#include "foundry-deploy-strategy.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_deploy_run (FoundryCommandLine *command_line,
                                const char * const *argv,
                                FoundryCliOptions  *options,
                                DexCancellable     *cancellable)
{
  g_autoptr(FoundryDeployStrategy) deploy_strategy = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  build_manager = foundry_context_dup_build_manager (context);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    goto handle_error;

  if (!(deploy_strategy = dex_await_object (foundry_deploy_strategy_new (pipeline), &error)))
    goto handle_error;

  if (!dex_await (foundry_deploy_strategy_deploy (deploy_strategy,
                                                  foundry_command_line_get_stdout (command_line),
                                                  cancellable),
                  &error))
    goto handle_error;

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_deploy (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "deploy"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_deploy_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("Deploy the application to active device"),
                                     });
}
