/* foundry-cli-builtin-device-list.c
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
#include "foundry-context.h"
#include "foundry-device.h"
#include "foundry-device-info.h"
#include "foundry-device-manager.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static void
foundry_cli_builtin_device_list_help (FoundryCommandLine *command_line)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  foundry_command_line_print (command_line, "Usage:\n");
  foundry_command_line_print (command_line, "  foundry device list [OPTIONS…]\n");
  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Options:\n");
  foundry_command_line_print (command_line, "      --help            Show help options\n");
  foundry_command_line_print (command_line, "  -f, --format=FORMAT   Output format (text, json)\n");
  foundry_command_line_print (command_line, "\n");
}

static int
foundry_cli_builtin_device_list_run (FoundryCommandLine *command_line,
                                     const char * const *argv,
                                     FoundryCliOptions  *options,
                                     DexCancellable     *cancellable)
{
  FoundryObjectSerializerFormat format;
  g_autoptr(FoundryDeviceManager) device_manager = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GListStore) infos = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;
  const char *format_arg;
  guint n_items;

  static const FoundryObjectSerializerEntry fields[] = {
    { "id", N_("ID") },
    { "active", N_("Active") },
    { "name", N_("Name") },
    { "chassis", N_("Chassis") },
    { "triplet", N_("System") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (foundry_cli_options_help (options))
    {
      foundry_cli_builtin_device_list_help (command_line);
      return EXIT_SUCCESS;
    }

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  device_manager = foundry_context_dup_device_manager (foundry);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (device_manager)), &error))
    goto handle_error;

  infos = g_list_store_new (FOUNDRY_TYPE_DEVICE_INFO);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (device_manager));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDevice) device = g_list_model_get_item (G_LIST_MODEL (device_manager), i);
      g_ptr_array_add (futures, foundry_device_load_info (device));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  for (guint i = 0; i < futures->len; i++)
    {
      g_autoptr(FoundryDeviceInfo) info = dex_await_object (dex_ref (futures->pdata[i]), NULL);

      if (info != NULL)
        g_list_store_append (infos, info);
    }

  format_arg = foundry_cli_options_get_string (options, "format");
  format = foundry_object_serializer_format_parse (format_arg);
  foundry_command_line_print_list (command_line, G_LIST_MODEL (infos), fields, format, G_TYPE_INVALID);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_device_list (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "device", "list"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         { "format", 'f', 0, G_OPTION_ARG_STRING, NULL, N_("Output format (text, json)"), N_("FORMAT") },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_device_list_run,
                                       .prepare = NULL,
                                       .complete = NULL,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("List available devices"),
                                     });
}
