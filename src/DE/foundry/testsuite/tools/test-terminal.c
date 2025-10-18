/* test-terminal.c
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
#include <foundry-gtk.h>

static GMainLoop *main_loop;

static DexFuture *
main_fiber (gpointer data)
{
  g_autoptr(FoundryTerminalLauncher) launcher = NULL;
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(VtePty) pty = NULL;
  GtkWidget *window;
  GtkWidget *scroll;
  GtkWidget *view;

  cancellable = dex_cancellable_new ();
  context = dex_await_object (foundry_context_new_for_user (cancellable), &error);
  g_assert_no_error (error);
  g_assert_nonnull (context);

  command = foundry_command_new (context);
  foundry_command_set_argv (command, FOUNDRY_STRV_INIT ("bash"));
  foundry_command_set_locality (command, FOUNDRY_COMMAND_LOCALITY_SUBPROCESS);

  launcher = foundry_terminal_launcher_new (command, NULL);

  window = g_object_new (GTK_TYPE_WINDOW,
                         "default-width", 800,
                         "default-height", 600,
                         NULL);
  scroll = g_object_new (GTK_TYPE_SCROLLED_WINDOW,
                         "hscrollbar-policy", GTK_POLICY_NEVER,
                         "propagate-natural-height", TRUE,
                         "propagate-natural-width", TRUE,
                         NULL);

  view = foundry_terminal_new ();
  vte_terminal_set_enable_fallback_scrolling (VTE_TERMINAL (view), FALSE);
  vte_terminal_set_scroll_unit_is_pixels (VTE_TERMINAL (view), TRUE);

  pty = vte_pty_new_sync (VTE_PTY_DEFAULT, NULL, NULL);
  vte_terminal_set_pty (VTE_TERMINAL (view), pty);

  if (!(subprocess = dex_await_object (foundry_terminal_launcher_run (launcher, vte_pty_get_fd (pty)), &error)))
    g_error ("%s", error->message);

  gtk_window_set_child (GTK_WINDOW (window), scroll);
  gtk_scrolled_window_set_child (GTK_SCROLLED_WINDOW (scroll), view);

  g_signal_connect_swapped (window,
                            "close-request",
                            G_CALLBACK (g_main_loop_quit),
                            main_loop);

  gtk_window_present (GTK_WINDOW (window));

  if (!dex_await (dex_subprocess_wait_check (subprocess), &error))
    g_error ("%s", error->message);

  g_main_loop_quit (main_loop);

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

  main_loop = g_main_loop_new (NULL, FALSE);
  dex_future_disown (dex_scheduler_spawn (NULL, 0, main_fiber, NULL, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
