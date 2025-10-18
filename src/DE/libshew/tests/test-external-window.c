/*
 * SPDX-FileCopyrightText: 2025 Florian Müllner <fmuellner@gnome.org>
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */
#include <gtk/gtk.h>
#include <shew.h>

static void
read_upto_one (GObject      *source,
               GAsyncResult *result,
               gpointer      user_data)
{
  GDataInputStream *out = G_DATA_INPUT_STREAM (source);
  g_autoptr (GError) error = NULL;
  char **out_str = user_data;

  *out_str = g_data_input_stream_read_upto_finish (out, result, NULL, &error);
  g_assert_no_error (error);

  g_main_context_wakeup (NULL);
}

static char *
client_get_handle (GSubprocess *client)
{
  g_autoptr (GDataInputStream) stream = NULL;
  char *handle = NULL;

  stream = g_data_input_stream_new (g_subprocess_get_stdout_pipe (client));
  g_data_input_stream_read_upto_async (stream,
                                       "\n", 1, 0, NULL, read_upto_one, &handle);

  while (!handle)
    g_main_context_iteration (NULL, TRUE);

  return handle;
}

static GSubprocess *
spawn_client (void)
{
  g_autofree char *client_path = NULL;

  client_path = g_test_build_filename (G_TEST_BUILT, "exported-client", NULL);
  return g_subprocess_new (G_SUBPROCESS_FLAGS_STDOUT_PIPE,
                           NULL,
                           client_path,
                           NULL);
}

static void
test_external_window (void)
{
  g_autoptr (GSubprocess) client = NULL;
  g_autoptr (ShewExternalWindow) external_window = NULL;
  GtkWindow *window = NULL;
  g_autofree char *handle = NULL;

  client = spawn_client ();
  handle = client_get_handle (client);
  g_assert_nonnull (handle);

  window = GTK_WINDOW (gtk_window_new ());
  gtk_widget_realize (GTK_WIDGET (window));

  external_window = shew_external_window_new_from_handle (handle);
  g_assert_nonnull (external_window);

  shew_external_window_set_parent_of (external_window,
                                      gtk_native_get_surface (GTK_NATIVE (window)));

  g_subprocess_force_exit (client);
}

static void
test_external_window_wayland (void)
{
  g_auto (GStrv) envp = NULL;

  if (g_test_subprocess ())
    {
      test_external_window ();
      return;
    }

  envp = g_get_environ ();
  envp = g_environ_setenv (g_steal_pointer (&envp), "GDK_BACKEND", "wayland", TRUE);
  g_test_trap_subprocess_with_envp (NULL, (const char * const *)envp, 0, G_TEST_SUBPROCESS_DEFAULT);
  g_test_trap_assert_passed ();
}

static void
test_external_window_x11 (void)
{
  g_auto (GStrv) envp = NULL;

  if (g_test_subprocess ())
    {
      test_external_window ();
      return;
    }

  envp = g_get_environ ();
  envp = g_environ_setenv (g_steal_pointer (&envp), "GDK_BACKEND", "x11", TRUE);
  g_test_trap_subprocess_with_envp (NULL, (const char * const *)envp, 0, G_TEST_SUBPROCESS_DEFAULT);
  g_test_trap_assert_passed ();
}

int
main (int    argc,
      char **argv)
{
  gtk_disable_portals ();
  gtk_test_init (&argc, &argv, NULL);

  g_test_add_func ("/Shew/ExternalWindow/wayland", test_external_window_wayland);
  g_test_add_func ("/Shew/ExternalWindow/x11", test_external_window_x11);

  return g_test_run ();
}
