/* list-palettes.c
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

static DexFuture *
main_fiber (gpointer user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GListModel) model = NULL;
  GMainLoop *main_loop = user_data;
  guint n_items;

  if (!(model = dex_await_object (foundry_terminal_list_palette_sets (), &error)))
    g_error ("Failed to list palettes: %s", error->message);

  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTerminalPaletteSet) set = g_list_model_get_item (model, i);
      g_autofree char *title = foundry_terminal_palette_set_dup_title (set);

      g_print ("%s\n", title);
    }

  g_main_loop_quit (main_loop);

  return NULL;
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(GMainLoop) main_loop = g_main_loop_new (NULL, FALSE);

  dex_init ();
  foundry_gtk_init ();

  dex_future_disown (dex_scheduler_spawn (NULL, 0, main_fiber,
                                          g_main_loop_ref (main_loop),
                                          (GDestroyNotify) g_main_loop_unref));

  g_main_loop_run (main_loop);

  return 0;
}
