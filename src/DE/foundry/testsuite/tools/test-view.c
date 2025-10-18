/* test-view.c
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

#include <gtk/gtk.h>
#include <gtksourceview/gtksource.h>

#include <foundry.h>
#include <foundry-gtk.h>

static GMainLoop *main_loop;

static DexFuture *
main_fiber (gpointer data)
{
  GFile *file = data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryTextManager) text_manager = NULL;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryOperation) operation = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *foundry_dir = NULL;
  GtkWidget *window;
  GtkWidget *scroll;
  GtkWidget *view;

  cancellable = dex_cancellable_new ();

  if ((foundry_dir = dex_await_string (foundry_context_discover (g_file_peek_path (file), cancellable), NULL)))
    context = dex_await_object (foundry_context_new (foundry_dir, NULL, 0, cancellable), &error);
  else
    context = dex_await_object (foundry_context_new_for_user (cancellable), &error);

  if (foundry_dir)
    g_print ("Foundry dir: %s\n", foundry_dir);

  g_assert_no_error (error);
  g_assert_nonnull (context);

  text_manager = foundry_context_dup_text_manager (context);
  operation = foundry_operation_new ();
  document = dex_await_object (foundry_text_manager_load (text_manager, file, operation, NULL), &error);
  g_assert_no_error (error);

  window = g_object_new (GTK_TYPE_WINDOW,
                         "default-width", 800,
                         "default-height", 600,
                         NULL);
  scroll = g_object_new (GTK_TYPE_SCROLLED_WINDOW, NULL);
  view = foundry_source_view_new (document);

  gtk_window_set_child (GTK_WINDOW (window), scroll);
  gtk_scrolled_window_set_child (GTK_SCROLLED_WINDOW (scroll), view);

  g_signal_connect_swapped (window,
                            "close-request",
                            G_CALLBACK (g_main_loop_quit),
                            main_loop);

  gtk_window_present (GTK_WINDOW (window));

  return dex_future_new_true ();
}

int
main (int    argc,
      char **argv)
{
  dex_init ();
  gtk_init ();

  dex_future_disown (foundry_init ());

  foundry_gtk_init ();
  gtk_source_init ();

  if (argc != 2)
    g_error ("usage: %s FILENAME", argv[0]);

  main_loop = g_main_loop_new (NULL, FALSE);
  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          main_fiber,
                                          g_file_new_for_commandline_arg (argv[1]),
                                          g_object_unref));
  g_main_loop_run (main_loop);

  return 0;
}
