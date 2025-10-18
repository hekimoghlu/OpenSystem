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
  g_autofree char *handle = NULL;

  handle = shew_window_exporter_export_finish (exporter, result, &error);
  g_assert_no_error (error);
  g_assert_nonnull (handle);
  g_print ("%s\n", handle);
}

int
main (int    argc,
      char **argv)
{
  g_autoptr (ShewWindowExporter) exporter = NULL;
  GtkWindow *window = NULL;

  gtk_disable_portals ();
  gtk_init ();

  window = GTK_WINDOW (gtk_window_new ());
  gtk_window_set_default_size (window, 100, 100);
  gtk_window_present (window);

  exporter = shew_window_exporter_new (window);
  shew_window_exporter_export (exporter, on_exported, NULL);

  while (TRUE)
    g_main_context_iteration (NULL, TRUE);

  return EXIT_SUCCESS;
}
