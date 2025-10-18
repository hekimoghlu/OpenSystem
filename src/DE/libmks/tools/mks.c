/* mks.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <unistd.h>

#include <gtk/gtk.h>
#include <libmks.h>

static GDBusConnection *
create_connection (int      argc,
                   char   **argv,
                   GError **error)
{
  if (argc < 2)
    return g_bus_get_sync (G_BUS_TYPE_SESSION, NULL, error);
  else
    return g_dbus_connection_new_for_address_sync (argv[1], G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT | G_DBUS_CONNECTION_FLAGS_MESSAGE_BUS_CONNECTION, NULL, NULL, error);
}

static gboolean
update_title_binding (GBinding     *binding,
                      const GValue *from_value,
                      GValue       *to_value,
                      gpointer      user_data)
{
  MksDisplay *display = user_data;
  GtkShortcutTrigger *trigger = mks_display_get_ungrab_trigger (display);
  g_autofree char *label = gtk_shortcut_trigger_to_label (trigger, gtk_widget_get_display (GTK_WIDGET (display)));

  if (g_value_get_boolean (from_value))
    g_value_take_string (to_value, g_strdup_printf ("MKS (%s to ungrab)", label));
  else
    g_value_set_static_string (to_value, "MKS");

  return TRUE;
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(GDBusConnection) connection = NULL;
  g_autoptr(MksSession) session = NULL;
  g_autoptr(MksScreen) screen = NULL;
  g_autoptr(GMainLoop) main_loop = NULL;
  g_autoptr(GError) error = NULL;
  GtkWindow *window;
  GtkWidget *display;
  GtkSettings *settings;
  GdkSurface *surface;

  gtk_init ();
  mks_init ();

  if (!(connection = create_connection (argc, argv, &error)))
    {
      g_printerr ("Failed to connect to D-Bus: %s\n", error->message);
      return EXIT_FAILURE;
    }

  main_loop = g_main_loop_new (NULL, FALSE);

  window = g_object_new (GTK_TYPE_WINDOW,
                         "default-width", 1280,
                         "default-height", 768,
                         "title", "Mouse, Keyboard, Screen",
                         NULL);

  settings = gtk_settings_get_default ();
  g_object_set (settings, "gtk-application-prefer-dark-theme", TRUE, NULL);

  display = mks_display_new ();
  gtk_window_set_child (window, display);
  g_signal_connect_swapped (window,
                            "close-request",
                            G_CALLBACK (g_main_loop_quit),
                            main_loop);

  if (!(session = mks_session_new_for_connection_sync (connection, NULL, &error)))
    {
      g_printerr ("Failed to create MksSession: %s\n", error->message);
      return EXIT_FAILURE;
    }

  if (!(screen = mks_session_ref_screen (session)))
    {
      g_printerr ("No screen attached to session!\n");
      return EXIT_FAILURE;
    }

  mks_display_set_screen (MKS_DISPLAY (display), screen);

  gtk_window_present (window);

  surface = gtk_native_get_surface (GTK_NATIVE (window));
  g_object_bind_property_full (surface, "shortcuts-inhibited",
                               window, "title",
                               G_BINDING_SYNC_CREATE,
                               update_title_binding, NULL, display, NULL);

  g_main_loop_run (main_loop);

  return EXIT_SUCCESS;
}
