/*
 * SPDX-FileCopyrightText: 2025 Florian Müllner <fmuellner@gnome.org>
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */
#include <gtk/gtk.h>
#include <shew.h>

static void
on_exported (GObject      *object,
             GAsyncResult *result,
             gpointer      user_data)
{
  ShewWindowExporter *exporter = (ShewWindowExporter *)object;
  g_autoptr (GError) error = NULL;
  char **out_handle = user_data;

  *out_handle = shew_window_exporter_export_finish (exporter, result, &error);
  g_assert_no_error (error);
}

static void
test_exporter (void)
{
  g_autoptr (ShewWindowExporter) exporter = NULL;
  GtkWindow *window = NULL;
  g_autofree char *handle = NULL;

  window = GTK_WINDOW (gtk_window_new ());
  gtk_window_present (window);

  exporter = shew_window_exporter_new (window);
  shew_window_exporter_export (exporter, on_exported, &handle);

  while (!handle)
    g_main_context_iteration (NULL, TRUE);

  g_assert_nonnull(handle);
  g_print ("%s\n", handle);
}

static void
test_exporter_wayland (void)
{
  g_auto (GStrv) envp = NULL;

  if (g_test_subprocess ())
    {
      test_exporter ();
      return;
    }

  envp = g_get_environ ();
  envp = g_environ_setenv (g_steal_pointer (&envp), "GDK_BACKEND", "wayland", TRUE);
  g_test_trap_subprocess_with_envp (NULL, (const char * const *)envp, 0, G_TEST_SUBPROCESS_DEFAULT);
  g_test_trap_assert_passed ();
  g_test_trap_assert_stdout ("wayland:*");
}

static void
test_exporter_x11 (void)
{
  g_auto (GStrv) envp = NULL;

  if (g_test_subprocess ())
    {
      test_exporter ();
      return;
    }

  envp = g_get_environ ();
  envp = g_environ_setenv (g_steal_pointer (&envp), "GDK_BACKEND", "x11", TRUE);
  g_test_trap_subprocess_with_envp (NULL, (const char * const *)envp, 0, G_TEST_SUBPROCESS_DEFAULT);
  g_test_trap_assert_passed ();
  g_test_trap_assert_stdout ("x11:*");
}

int
main (int    argc,
      char **argv)
{
  gtk_disable_portals ();
  gtk_test_init (&argc, &argv, NULL);

  g_test_add_func ("/Shew/Exporter/wayland", test_exporter_wayland);
  g_test_add_func ("/Shew/Exporter/x11", test_exporter_x11);

  return g_test_run ();
}
