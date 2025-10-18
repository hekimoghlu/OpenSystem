/* foundry-cli-builtin-doc-bundle-install.c
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
#include "foundry-command-line-private.h"
#include "foundry-context.h"
#include "foundry-documentation-bundle.h"
#include "foundry-documentation-manager.h"
#include "foundry-operation-manager.h"
#include "foundry-operation.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static void
redraw_progress (FoundryOperation   *operation,
                 GParamSpec         *pspec,
                 FoundryCommandLine *command_line)
{
  g_autofree char *title = NULL;
  g_autofree char *subtitle = NULL;
  double fraction;

  g_assert (FOUNDRY_IS_OPERATION (operation));
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  title = foundry_operation_dup_title (operation);
  subtitle = foundry_operation_dup_subtitle (operation);
  fraction = foundry_operation_get_progress (operation);

  foundry_command_line_set_progress (command_line, fraction);
  foundry_command_line_clear_line (command_line);
  foundry_command_line_print (command_line,
                              "%s: %s%s%u%%",
                              title,
                              subtitle ? subtitle : "",
                              subtitle ? ": " : "",
                              (guint)(100*fraction));
}

static char **
foundry_cli_builtin_doc_bundle_install_complete (FoundryCommandLine *command_line,
                                                 const char         *command,
                                                 const GOptionEntry *entry,
                                                 FoundryCliOptions  *options,
                                                 const char * const *argv,
                                                 const char         *current)
{
  g_autoptr(FoundryDocumentationManager) documentation_manager = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GListModel) bundles = NULL;
  g_autoptr(GStrvBuilder) builder = g_strv_builder_new ();
  g_autoptr(GError) error = NULL;
  guint n_bundles;

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    return NULL;

  documentation_manager = foundry_context_dup_documentation_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (documentation_manager)), &error))
    return NULL;

  if (!(bundles = dex_await_object (foundry_documentation_manager_list_bundles (documentation_manager), &error)))
    return NULL;

  n_bundles = g_list_model_get_n_items (bundles);

  for (guint i = 0; i < n_bundles; i++)
    {
      g_autoptr(FoundryDocumentationBundle) bundle = g_list_model_get_item (bundles, i);
      g_autofree char *id = foundry_documentation_bundle_dup_id (bundle);
      g_autofree char *spaced = g_strdup_printf ("%s ", id);

      if (current == NULL ||
          g_str_has_prefix (spaced, current))
        g_strv_builder_add (builder, spaced);
    }

  return g_strv_builder_end (builder);
}

static int
foundry_cli_builtin_doc_bundle_install_run (FoundryCommandLine *command_line,
                                            const char * const *argv,
                                            FoundryCliOptions  *options,
                                            DexCancellable     *cancellable)
{
  g_autoptr(FoundryDocumentationManager) documentation_manager = NULL;
  g_autoptr(FoundryOperationManager) operation_manager = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GListModel) bundles = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *bundle_id = NULL;
  guint n_bundles;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!(bundle_id = g_strdup (argv[1])))
    {
      foundry_command_line_printerr (command_line, "usage: foundry doc bundle install BUNDLE_ID\n");
      return EXIT_FAILURE;
    }

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  documentation_manager = foundry_context_dup_documentation_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (documentation_manager)), &error))
    goto handle_error;

  operation_manager = foundry_context_dup_operation_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (operation_manager)), &error))
    goto handle_error;

  if (!(bundles = dex_await_object (foundry_documentation_manager_list_bundles (documentation_manager), &error)))
    goto handle_error;

  n_bundles = g_list_model_get_n_items (bundles);

  for (guint i = 0; i < n_bundles; i++)
    {
      g_autoptr(FoundryDocumentationBundle) bundle = g_list_model_get_item (bundles, i);
      g_autofree char *id = foundry_documentation_bundle_dup_id (bundle);

      if (g_strcmp0 (id, bundle_id) == 0)
        {
          g_autoptr(FoundryOperation) operation = NULL;

          operation = foundry_operation_manager_begin (operation_manager, _("Installing Documentation"));

          g_signal_connect_object (operation,
                                   "notify",
                                   G_CALLBACK (redraw_progress),
                                   command_line,
                                   0);

          dex_await (foundry_documentation_bundle_install (bundle, operation, cancellable), &error);
          foundry_command_line_print (command_line, "\n");

          if (error != NULL)
            goto handle_error;

          break;
        }
    }

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_doc_bundle_install (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "doc", "bundle", "install"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_doc_bundle_install_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_doc_bundle_install_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("Install a documentation bundle"),
                                     });
}
