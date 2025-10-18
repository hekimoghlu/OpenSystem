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

#include "ctags/plugin-ctags-file.h"

static char **real_argv;
static int real_argc;

static DexFuture *
load_fiber (gpointer data)
{
  GMainLoop *main_loop = data;

  if (real_argc > 1)
    {
      g_autoptr(GFile) file = g_file_new_for_path (real_argv[1]);
      g_autoptr(GError) error = NULL;
      g_autoptr(PluginCtagsFile) ctags = dex_await_object (plugin_ctags_file_new (file), &error);

      g_assert_no_error (error);

      if (real_argc < 3)
        {
          gsize size = plugin_ctags_file_get_size (ctags);

          for (gsize j = 0; j < size; j++)
            {
              g_autofree char *name = plugin_ctags_file_dup_name (ctags, j);
              g_autofree char *path = plugin_ctags_file_dup_path (ctags, j);
              g_autofree char *pattern = plugin_ctags_file_dup_pattern (ctags, j);
              g_autofree char *keyval = plugin_ctags_file_dup_keyval (ctags, j);

              g_print ("%u: %c: `%s` `%s` `%s` `%s`\n",
                       (guint)j,
                       plugin_ctags_file_get_kind (ctags, j),
                       name, path, pattern, keyval);
            }
        }
      else
        {
          g_autoptr(GListModel) results = dex_await_object (plugin_ctags_file_match (ctags, real_argv[2]), &error);
          guint n_items;

          g_assert_no_error (error);
          g_assert_nonnull (results);
          g_assert_true (G_IS_LIST_MODEL (results));

          n_items = g_list_model_get_n_items (G_LIST_MODEL (results));

          for (guint i = 0; i < n_items; i++)
            {
              g_autoptr(FoundryCompletionProposal) proposal = g_list_model_get_item (results, i);
              g_autofree char *typed_text = NULL;

              g_assert_nonnull (proposal);
              g_assert_true (FOUNDRY_IS_COMPLETION_PROPOSAL (proposal));

              typed_text = foundry_completion_proposal_dup_typed_text (proposal);

              g_print ("%s\n", typed_text);
            }
        }
    }

  g_main_loop_quit (main_loop);

  return dex_future_new_true ();
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(GMainLoop) main_loop = g_main_loop_new (NULL, FALSE);

  real_argv = argv;
  real_argc = argc;

  dex_init ();

  dex_future_disown (dex_scheduler_spawn (NULL, 0, load_fiber, main_loop, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
