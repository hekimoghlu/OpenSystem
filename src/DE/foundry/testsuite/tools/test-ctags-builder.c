/* test-ctags.c
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

#include <foundry.h>

#include "ctags/plugin-ctags-builder.h"
#include "ctags/plugin-ctags-file.h"

static char *dest_filename;
static char *dir_filename;

static DexFuture *
load_fiber (gpointer data)
{
  GMainLoop *main_loop = data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) destination = g_file_new_for_path (dest_filename);
  g_autoptr(GFile) dir = g_file_new_for_path (dir_filename);
  g_autoptr(PluginCtagsBuilder) builder = plugin_ctags_builder_new (destination);
  g_autoptr(GFileEnumerator) enumerator = dex_await_object (dex_file_enumerate_children (dir, G_FILE_ATTRIBUTE_STANDARD_NAME, 0, 0), &error);
  gboolean rval;

  g_assert_no_error (error);
  g_assert_nonnull (enumerator);

  for (;;)
    {
      g_autolist(GFileInfo) files = dex_await_boxed (dex_file_enumerator_next_files (enumerator, 100, 0), &error);

      g_assert_no_error (error);

      if (files == NULL)
        break;

      for (const GList *iter = files; iter; iter = iter->next)
        {
          g_autoptr(GFile) file = g_file_enumerator_get_child (enumerator, iter->data);

          plugin_ctags_builder_add_file (builder, file);
        }
    }

  rval = dex_await (plugin_ctags_builder_build (builder), &error);

  g_assert_no_error (error);
  g_assert_true (rval);

  g_main_loop_quit (main_loop);

  return dex_future_new_true ();
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(GMainLoop) main_loop = g_main_loop_new (NULL, FALSE);

  if (argc != 3)
    {
      g_printerr ("usage: %s OUT_FILE DIRECTORY\n", argv[0]);
      return 1;
    }

  dest_filename = argv[1];
  dir_filename = argv[2];

  dex_init ();

  dex_future_disown (dex_scheduler_spawn (NULL, 0, load_fiber, main_loop, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
