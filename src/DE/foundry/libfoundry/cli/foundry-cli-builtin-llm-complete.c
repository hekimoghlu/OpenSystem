/* foundry-cli-builtin-llm-complete.c
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
#include "foundry-llm-completion.h"
#include "foundry-llm-completion-chunk.h"
#include "foundry-llm-manager.h"
#include "foundry-llm-model.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static int
foundry_cli_builtin_llm_complete_run (FoundryCommandLine *command_line,
                                      const char * const *argv,
                                      FoundryCliOptions  *options,
                                      DexCancellable     *cancellable)
{
  g_autoptr(FoundryLlmCompletion) completion = NULL;
  g_autoptr(FoundryLlmManager) llm_manager = NULL;
  g_autoptr(FoundryLlmModel) model = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(GError) error = NULL;
  const char *name;
  const char *prompt;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (g_strv_length ((char **)argv) != 3)
    {
      foundry_command_line_printerr (command_line, "usage: %s MODEL \"PROMPT\"\n", argv[0]);
      return EXIT_FAILURE;
    }

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  name = argv[1];
  prompt = argv[2];

  llm_manager = foundry_context_dup_llm_manager (foundry);

  if (!(model = dex_await_object (foundry_llm_manager_find_model (llm_manager, name), &error)))
    goto handle_error;

  if (!(completion = dex_await_object (foundry_llm_model_complete (model,
                                                                   FOUNDRY_STRV_INIT ("user"),
                                                                   FOUNDRY_STRV_INIT (prompt)),
                                       &error)))
    goto handle_error;

  for (;;)
    {
      g_autoptr(FoundryLlmCompletionChunk) chunk = NULL;
      g_autofree char *text = NULL;

      if (!(chunk = dex_await_object (foundry_llm_completion_next_chunk (completion), &error)))
        goto handle_error;

      if ((text = foundry_llm_completion_chunk_dup_text (chunk)))
        foundry_command_line_print (command_line, "%s", text);

      if (foundry_llm_completion_chunk_is_done (chunk))
        break;
    }

  foundry_command_line_print (command_line, "\n");

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_llm_complete (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "llm", "complete"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_llm_complete_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("MODEL - Complete using a LLM"),
                                     });
}
