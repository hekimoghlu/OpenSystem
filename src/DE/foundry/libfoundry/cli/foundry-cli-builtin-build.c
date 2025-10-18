/* foundry-cli-builtin-build.c
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
foundry_cli_builtin_build_error (FoundryCommandLine *command_line,
                                 const GError       *error)
{
  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

static int
foundry_cli_builtin_purge_run (FoundryCommandLine *command_line,
                               const char * const *argv,
                               FoundryCliOptions  *options,
                               DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_purge (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_MASK (-1),
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  return EXIT_SUCCESS;
}

static int
foundry_cli_builtin_clean_run (FoundryCommandLine *command_line,
                               const char * const *argv,
                               FoundryCliOptions  *options,
                               DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_clean (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  return EXIT_SUCCESS;
}

static int
foundry_cli_builtin_build_run (FoundryCommandLine *command_line,
                               const char * const *argv,
                               FoundryCliOptions  *options,
                               DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  return EXIT_SUCCESS;
}

static int
foundry_cli_builtin_rebuild_run (FoundryCommandLine *command_line,
                                 const char * const *argv,
                                 FoundryCliOptions  *options,
                                 DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_purge (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_MASK (-1),
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  g_clear_object (&progress);

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  return EXIT_SUCCESS;
}

static int
foundry_cli_builtin_install_run (FoundryCommandLine *command_line,
                                 const char * const *argv,
                                 FoundryCliOptions  *options,
                                 DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_INSTALL,
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  return EXIT_SUCCESS;
}

static int
foundry_cli_builtin_configure_run (FoundryCommandLine *command_line,
                                   const char * const *argv,
                                   FoundryCliOptions  *options,
                                   DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE,
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  return EXIT_SUCCESS;
}

static int
foundry_cli_builtin_export_run (FoundryCommandLine *command_line,
                                const char * const *argv,
                                FoundryCliOptions  *options,
                                DexCancellable     *cancellable)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GListModel) artifacts = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *existing = NULL;
  guint n_artifacts;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  build_manager = foundry_context_dup_build_manager (foundry);

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return foundry_cli_builtin_build_error (command_line, error);

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_EXPORT,
                                           foundry_command_line_get_stdout (command_line),
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return foundry_cli_builtin_build_error (command_line, error);

  artifacts = foundry_build_progress_list_artifacts (progress);
  n_artifacts = g_list_model_get_n_items (artifacts);

  if (n_artifacts == 0)
    return EXIT_SUCCESS;

  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Artifacts:\n");

  for (guint i = 0; i < n_artifacts; i++)
    {
      g_autoptr(GFile) artifact = g_list_model_get_item (artifacts, i);
      g_autofree char *uri = g_file_get_uri (artifact);

      foundry_command_line_print (command_line, "  %s\n", uri);
    }

  foundry_command_line_print (command_line, "\n");

  return EXIT_SUCCESS;
}

typedef struct _CommandAlias
{
  const char * const *alias;
  int (*command) (FoundryCommandLine *, const char * const *, FoundryCliOptions *, DexCancellable*);
} CommandAlias;

void
foundry_cli_builtin_build (FoundryCliCommandTree *tree)
{
  const CommandAlias aliases[] = {
    { FOUNDRY_STRV_INIT ("foundry", "build"), foundry_cli_builtin_build_run },
    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "build"), foundry_cli_builtin_build_run },

    { FOUNDRY_STRV_INIT ("foundry", "rebuild"), foundry_cli_builtin_rebuild_run },
    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "rebuild"), foundry_cli_builtin_rebuild_run },

    { FOUNDRY_STRV_INIT ("foundry", "clean"), foundry_cli_builtin_clean_run },
    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "clean"), foundry_cli_builtin_clean_run },

    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "purge"), foundry_cli_builtin_purge_run },

    { FOUNDRY_STRV_INIT ("foundry", "install"), foundry_cli_builtin_install_run },
    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "install"), foundry_cli_builtin_install_run },

    { FOUNDRY_STRV_INIT ("foundry", "export"), foundry_cli_builtin_export_run },
    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "export"), foundry_cli_builtin_export_run },

    { FOUNDRY_STRV_INIT ("foundry", "pipeline", "configure"), foundry_cli_builtin_configure_run },
  };

  for (guint i = 0; i < G_N_ELEMENTS (aliases); i++)
    foundry_cli_command_tree_register (tree,
                                       aliases[i].alias,
                                       &(FoundryCliCommand) {
                                         .options = (GOptionEntry[]) {
                                           { "help", 0, 0, G_OPTION_ARG_NONE },
                                           {0}
                                         },
                                         .run = aliases[i].command,
                                         .prepare = NULL,
                                         .complete = NULL,
                                         .gettext_package = GETTEXT_PACKAGE,
                                         .description = N_("Build the project"),
                                       });
}
