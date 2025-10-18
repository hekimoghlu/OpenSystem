/* foundry-cli-builtin-vcs-blame.c
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

#include "line-reader-private.h"

#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command-line.h"
#include "foundry-context.h"
#include "foundry-vcs.h"
#include "foundry-vcs-blame.h"
#include "foundry-vcs-file.h"
#include "foundry-vcs-manager.h"
#include "foundry-vcs-signature.h"
#include "foundry-service.h"

static int
foundry_cli_builtin_vcs_blame_run (FoundryCommandLine *command_line,
                                   const char * const *argv,
                                   FoundryCliOptions  *options,
                                   DexCancellable     *cancellable)
{
  g_autoptr(FoundryVcsManager) vcs_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryVcsBlame) blame = NULL;
  g_autoptr(FoundryVcsFile) vcs_file = NULL;
  g_autoptr(FoundryVcs) vcs = NULL;
  g_autoptr(GString) str = NULL;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) file = NULL;
  g_autofree char *dir = NULL;
  LineReader reader;
  guint n_lines;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (options != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (g_strv_length ((char **)argv) < 2)
    {
      foundry_command_line_printerr (command_line, "usage: %s FILE\n", argv[0]);
      return EXIT_FAILURE;
    }

  dir = foundry_command_line_get_directory (command_line);
  file = g_file_new_build_filename (dir, argv[1], NULL);

  if (!(bytes = dex_await_boxed (dex_file_load_contents_bytes (file), &error)))
    goto handle_error;

  line_reader_init_from_bytes (&reader, bytes);

  if (!(context = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  vcs_manager = foundry_context_dup_vcs_manager (context);
  if (!dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (vcs_manager)), &error))
    goto handle_error;

  if (!(vcs = foundry_vcs_manager_dup_vcs (vcs_manager)))
    {
      foundry_command_line_printerr (command_line, "No VCS in use.\n");
      return EXIT_FAILURE;
    }

  if (!(vcs_file = dex_await_object (foundry_vcs_find_file (vcs, file), &error)) ||
      !(blame = dex_await_object (foundry_vcs_blame (vcs, vcs_file, bytes), &error)))
    goto handle_error;

  n_lines = foundry_vcs_blame_get_n_lines (blame);
  str = g_string_new (NULL);

  for (guint i = 0; i < n_lines; i++)
    {
      g_autoptr(FoundryVcsSignature) signature = foundry_vcs_blame_query_line (blame, i);
      g_autoptr(GDateTime) when = NULL;
      g_autofree char *name = NULL;
      g_autofree char *when_str = NULL;
      const char *line;
      gsize line_len;

      if (signature != NULL)
        {
          name = foundry_vcs_signature_dup_name (signature);
          when = foundry_vcs_signature_dup_when (signature);

          if (when != NULL)
            when_str = g_date_time_format (when, "%x %X %:z");
        }

      g_string_append_c (str, '(');
      g_string_append_printf (str, "%30s", name ? name : "");
      g_string_append_c (str, ' ');
      g_string_append_printf (str, "%20s", when_str ? when_str : "");
      g_string_append_c (str, ' ');
      g_string_append_printf (str, "%5d", i + 1);
      g_string_append_c (str, ')');
      g_string_append_c (str, ' ');
      if ((line = line_reader_next (&reader, &line_len)))
        g_string_append_len (str, line, line_len);
      g_string_append_c (str, '\n');

      foundry_command_line_print (command_line, "%s", str->str);

      g_string_truncate (str, 0);
    }


  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

static char **
foundry_cli_builtin_vcs_blame_complete (FoundryCommandLine *command_line,
                                        const char         *command,
                                        const GOptionEntry *entry,
                                        FoundryCliOptions  *options,
                                        const char * const *argv,
                                        const char         *current)
{
  return g_strdupv ((char **)FOUNDRY_STRV_INIT ("__FOUNDRY_FILE"));
}

void
foundry_cli_builtin_vcs_blame (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "vcs", "blame"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_vcs_blame_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_vcs_blame_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("FILE - Print commit blame information"),
                                     });
}
