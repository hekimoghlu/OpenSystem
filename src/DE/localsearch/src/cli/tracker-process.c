/*
 * Copyright (C) 2010, Nokia <ivan.frade@nokia.com>
 * Copyright (C) 2014, Lanedo <martyn@lanedo.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */

#include "config-miners.h"

#include <stdlib.h>
#include <errno.h>

#include <glib.h>
#include <glib/gi18n.h>

#include "tracker-process.h"

static pid_t
get_pid_for_service (GDBusConnection *connection,
                     const gchar     *name)
{
	GDBusMessage *message, *reply;
	GVariant *variant;
	guint32 process_id;

	message = g_dbus_message_new_method_call ("org.freedesktop.DBus",
	                                          "/org/freedesktop/DBus",
	                                          "org.freedesktop.DBus",
	                                          "GetConnectionUnixProcessID");
	g_dbus_message_set_body (message,
	                         g_variant_new ("(s)", name));
	reply = g_dbus_connection_send_message_with_reply_sync (connection,
	                                                        message,
	                                                        G_DBUS_SEND_MESSAGE_FLAGS_NONE,
	                                                        -1,
	                                                        NULL,
	                                                        NULL,
	                                                        NULL);
	g_object_unref (message);

	if (!reply)
		return -1;

	if (g_dbus_message_get_error_name (reply)) {
		g_object_unref (reply);
		return -1;
	}

	variant = g_dbus_message_get_body (reply);
	g_variant_get (variant, "(u)", &process_id);
	g_object_unref (reply);

	return (pid_t) process_id;
}

pid_t
tracker_process_find (void)
{
	g_autoptr (GDBusConnection) connection = NULL;

	connection = g_bus_get_sync (G_BUS_TYPE_SESSION, NULL, NULL);

	return get_pid_for_service (connection, "org.freedesktop.LocalSearch3");
}

gint
tracker_process_stop (gint signal_id)
{
	pid_t indexer_pid;

	indexer_pid = tracker_process_find ();
	if (indexer_pid < 0)
		return 0;

	if (kill (indexer_pid, signal_id) == -1) {
		g_printerr ("%s: %s\n", _("Could not terminate indexer"),
		            g_strerror (errno));
	} else {
		g_print ("%s\n", _("Indexer process terminated"));
	}

	return 0;
}
