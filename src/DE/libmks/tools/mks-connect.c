/*
 * mks-connect.c
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

#include <locale.h>
#include <unistd.h>

#include <glib/gi18n.h>
#include <gtk/gtk.h>
#include <libmks.h>

static void
print_device_info (MksDevice *device,
                   guint      depth)
{
  if (device == NULL)
    return;

  for (guint i = 0; i < depth; i++)
    g_print ("  ");

  g_print ("- %s(name=\"%s\"",
           G_OBJECT_TYPE_NAME (device),
           mks_device_get_name (device) ?: "");
  if (MKS_IS_SCREEN (device))
    g_print (", address=\"%s\", number=%u, width=%u, height=%u",
             mks_screen_get_device_address (MKS_SCREEN (device)),
             mks_screen_get_number (MKS_SCREEN (device)),
             mks_screen_get_width (MKS_SCREEN (device)),
             mks_screen_get_height (MKS_SCREEN (device)));
  else if (MKS_IS_KEYBOARD (device))
    g_print (", modifiers=0x%x",
             mks_keyboard_get_modifiers (MKS_KEYBOARD (device)));
  else if (MKS_IS_MOUSE (device))
    g_print (", is-absolute=%u",
             mks_mouse_get_is_absolute (MKS_MOUSE (device)));
  else if (MKS_IS_TOUCHABLE (device))
    g_print (", max-slots=%u",
             mks_touchable_get_max_slots (MKS_TOUCHABLE (device)));
  g_print (")\n");

  if (MKS_IS_SCREEN (device))
    {
      MksScreen *screen = MKS_SCREEN (device);
      MksKeyboard *keyboard = mks_screen_get_keyboard (screen);
      MksMouse *mouse = mks_screen_get_mouse (screen);
      MksTouchable *touchable = mks_screen_get_touchable (screen);

      print_device_info (MKS_DEVICE (keyboard), depth+1);
      print_device_info (MKS_DEVICE (mouse), depth+1);
      print_device_info (MKS_DEVICE (touchable), depth+1);
    }
}

int
main (int argc,
      char *argv[])
{
  g_autoptr(GOptionContext) context = g_option_context_new ("DBUS_ADDRESS - Connect to QEMU at DBUS_ADDRESS");
  g_autoptr(GDBusConnection) connection = NULL;
  g_autoptr(MksSession) session = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GVariant) variant = NULL;
  g_autoptr(GDBusProxy) proxy = NULL;
  g_autofree const char **queued_owners = NULL;
  gsize n_queued_owners;
  GListModel *devices = NULL;
  guint n_items;

  setlocale (LC_ALL, "");
  bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");

  gtk_init ();
  mks_init ();

  if (!g_option_context_parse (context, &argc, &argv, &error))
    {
      g_printerr ("%s\n", error->message);
      return EXIT_FAILURE;
    }

  if (argc < 2)
    connection = g_bus_get_sync (G_BUS_TYPE_SESSION, NULL, &error);
  else
    connection = g_dbus_connection_new_for_address_sync (argv[1], 0, NULL, NULL, &error);

  if (connection == NULL)
    {
      g_printerr ("Failed to connect to D-Bus: %s\n",
                  error->message);
      return EXIT_FAILURE;
    }

  proxy = g_dbus_proxy_new_sync (connection,
                                 G_DBUS_PROXY_FLAGS_NONE,
                                 NULL,
                                 "org.freedesktop.DBus",
                                 "/org/freedesktop/DBus",
                                 "org.freedesktop.DBus",
                                 NULL,
                                 &error);

  if (proxy == NULL)
    {
      g_printerr ("Failed to connect to `org.freedesktop.DBus`: %s\n",
                  error->message);
      return EXIT_FAILURE;
    }
  variant = g_dbus_proxy_call_sync (proxy, "ListQueuedOwners",
                                    g_variant_new ("(s)", "org.qemu"), G_DBUS_CALL_FLAGS_NONE,
                                    -1, NULL, &error);
  if (error != NULL)
    {
      g_printerr ("Failed to ListQueuedOwners: %s\n",
                  error->message);
      return EXIT_FAILURE;
    }

  queued_owners  = g_variant_get_strv (g_variant_get_child_value (variant, 0),
                                       &n_queued_owners);

  for (guint i = 0; i < n_queued_owners; i++)
    {
      if (!(session = mks_session_new_for_connection_with_name_sync (connection, queued_owners[i], NULL, &error)))
        {
          g_printerr ("Failed to create MksSession: %s\n",
                      error->message);
          return EXIT_FAILURE;
        }

      g_print ("Session(uuid=\"%s\", name=\"%s\", bus-name=%s)\n",
              mks_session_get_uuid (session),
              mks_session_get_name (session),
              mks_session_get_bus_name (session));

      devices = mks_session_get_devices (session);
      n_items = g_list_model_get_n_items (devices);

      for (guint j = 0; j < n_items; j++)
        {
          g_autoptr(MksDevice) device = g_list_model_get_item (devices, j);
          print_device_info (device, 1);
        }
    }
  return EXIT_SUCCESS;
}
