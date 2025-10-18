/* list-tweaks.c
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

#include <foundry-gtk.h>

static GMainLoop *main_loop;

static void
print_tree (FoundryTweakManager *manager,
            const char          *the_path)
{
  g_autoptr(GListModel) model = NULL;
  g_autoptr(GError) error = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_TWEAK_MANAGER (manager));
  g_assert (the_path != NULL);

  if (!(model = dex_await_object (foundry_tweak_manager_list_children (manager, the_path), &error)))
    g_error ("%s", error->message);

  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTweak) tweak = g_list_model_get_item (model, i);
      g_autofree char *path = foundry_tweak_dup_path (tweak);
      g_autofree char *title = foundry_tweak_dup_title (tweak);

      g_print ("%s [Title: %s]\n", path, title ? title : "");

      print_tree (manager, path);
    }
}

static DexFuture *
main_fiber (gpointer user_data)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryTweakManager) manager = NULL;
  g_autoptr(GListModel) app = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *path = NULL;
  const char *dirpath = ".";

  dex_await (foundry_init (), NULL);

  if (!(path = dex_await_string (foundry_context_discover (dirpath, NULL), &error)))
    g_error ("%s", error->message);

  if (!(context = dex_await_object (foundry_context_new (path, dirpath, FOUNDRY_CONTEXT_FLAGS_NONE, NULL), &error)))
    g_error ("%s", error->message);

  manager = foundry_context_dup_tweak_manager (context);

  print_tree (manager, "/app/");
  print_tree (manager, "/project/");
  print_tree (manager, "/user/");

  g_main_loop_quit (main_loop);

  return NULL;
}

int
main (int argc,
      char *argv[])
{
  foundry_gtk_init ();

  main_loop = g_main_loop_new (NULL, FALSE);
  dex_future_disown (dex_scheduler_spawn (NULL, 0, main_fiber, NULL, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
