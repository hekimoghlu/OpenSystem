/*
 * mks-util.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
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

#include <glib/gstdio.h>
#include <sys/socket.h>

#include "mks-util-private.h"

static GSettings *mouse_settings;
static GSettings *touchpad_settings;
static gsize initialized;

static GSettings *
load_gsettings (const char *schema_id)
{
  GSettingsSchemaSource *source = g_settings_schema_source_get_default ();
  g_autoptr(GSettingsSchema) schema = g_settings_schema_source_lookup (source, schema_id, TRUE);

  if (schema != NULL)
    return g_settings_new (schema_id);

  return NULL;
}

static void
_mks_util_init (void)
{
  if (g_once_init_enter (&initialized))
    {
      mouse_settings = load_gsettings ("org.gnome.desktop.peripherals.mouse");
      touchpad_settings = load_gsettings ("org.gnome.desktop.peripherals.touchpad");
      g_once_init_leave (&initialized, TRUE);
    }
}

/* This is abstracted in a way that as soon as GdkEvent contains enough
 * information to know if the GdkScrollEvent contains inverted axis
 * directoin we can use that instead of checking the GSetting.
 *
 * TODO: This won't handle Flatpak because we won't have access to the
 * host setting for the GSetting. Additionally, it won't work with jhbuild
 * for the same reasons (likely using alternate GSettings/dconf).
 *
 * But this is better than nothing for the time being and provides an
 * abstraction point once support for wayland!183 lands.
 */
gboolean
mks_scroll_event_is_inverted (GdkEvent *event)
{
  GdkScrollUnit unit;

  g_return_val_if_fail (gdk_event_get_event_type (event) == GDK_SCROLL, FALSE);

  _mks_util_init ();

  if (mouse_settings == NULL || touchpad_settings == NULL)
    return FALSE;

  unit = gdk_scroll_event_get_unit (event);

  switch (unit)
    {
    case GDK_SCROLL_UNIT_WHEEL:
      return g_settings_get_boolean (mouse_settings, "natural-scroll");

    case GDK_SCROLL_UNIT_SURFACE:
      return g_settings_get_boolean (touchpad_settings, "natural-scroll");

    default:
      return FALSE;
    }
}

gboolean
mks_socketpair_create (int     *us,
                       int     *them,
                       GError **error)
{
  int fds[2];
  int rv;

  rv = socketpair (AF_UNIX, SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC, 0, fds);

  if (rv != 0)
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
      return FALSE;
    }

  *us = fds[0];
  *them = fds[1];

  return TRUE;
}

static void
fdptr_clear (gpointer data)
{
  int *fdptr = data;
  if (*fdptr != -1)
    close (*fdptr);
  *fdptr = -1;
  g_free (fdptr);
}

static void
mks_socketpair_connection_cb (GObject      *object,
                              GAsyncResult *result,
                              gpointer      user_data)
{
  g_autoptr(GTask) task = user_data;
  g_autoptr(GDBusConnection) ret = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (G_IS_TASK (task));

  if (!(ret = g_dbus_connection_new_finish (result, &error)))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_task_return_pointer (task, g_steal_pointer (&ret), g_object_unref);
}

void
mks_socketpair_connection_new (GDBusConnectionFlags  flags,
                               GCancellable         *cancellable,
                               GAsyncReadyCallback   callback,
                               gpointer              user_data)
{
  g_autoptr(GSocketConnection) io_stream = NULL;
  g_autoptr(GSocket) socket = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GTask) task = NULL;
  g_autofd int us = -1;
  g_autofd int them = -1;
  int *fdptr;

  g_return_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable));

  task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_source_tag (task, mks_socketpair_connection_new);
  g_task_set_task_data (task, GINT_TO_POINTER (-1), NULL);

  if (!mks_socketpair_create (&us, &them, &error) ||
      !(socket = g_socket_new_from_fd (us, &error)))
    {
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  io_stream = g_socket_connection_factory_create_connection (socket);
  fdptr = g_memdup2 (&them, sizeof them);
  g_task_set_task_data (task, fdptr, fdptr_clear);

  us = -1;
  them = -1;

  g_dbus_connection_new (G_IO_STREAM (io_stream),
                         NULL,
                         flags | G_DBUS_CONNECTION_FLAGS_DELAY_MESSAGE_PROCESSING,
                         NULL,
                         cancellable,
                         mks_socketpair_connection_cb,
                         g_steal_pointer (&task));

}

/**
 * mks_socketpair_connection_new_finish:
 * @result: a #GAsyncResult
 * @peer_fd: (out): a location for a socketpair file-descriptor
 * @error: (out): a location for a #GError, or %NULL
 *
 * Completes the asynchronous request to create a socketpair()-based
 * D-Bus connection.
 *
 * Returns: (transfer full): a #GDBusConnection and @peer_fd is set
 *   if successful; otherwise %NULL and @error is set.
 */
GDBusConnection *
mks_socketpair_connection_new_finish (GAsyncResult  *result,
                                      int           *peer_fd,
                                      GError       **error)
{
  GDBusConnection *ret;
  int *fdptr;

  g_return_val_if_fail (G_IS_TASK (result), NULL);
  g_return_val_if_fail (peer_fd != NULL, NULL);

  fdptr = g_task_get_task_data (G_TASK (result));

  if ((ret = g_task_propagate_pointer (G_TASK (result), error)))
    *peer_fd = g_steal_fd (fdptr);

  return ret;
}
