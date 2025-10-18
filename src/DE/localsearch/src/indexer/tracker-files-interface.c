/*
 * Copyright (C) 2023 Red Hat Inc.

 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include "tracker-files-interface.h"

#include <tracker-common.h>

#include <gio/gunixfdlist.h>
#include <sys/mman.h>

struct _TrackerFilesInterface
{
	GObject parent_instance;
	GDBusConnection *connection;
	GSettings *settings;
	GVariant *priority_graphs;
#ifdef HAVE_POWER
	TrackerPower *power;
#endif
	guint object_id;
	int fd;
};

enum {
	PROP_0,
	PROP_CONNECTION,
	PROP_PERSISTENCE_FD,
	N_PROPS,
};

static GParamSpec *props[N_PROPS] = { 0, };

static const gchar *introspection_xml =
	"<node>"
	"  <interface name='org.freedesktop.Tracker3.Files'>"
	"    <property name='ExtractorConfig' type='a{sv}' access='read' />"
	"    <method name='GetPersistenceStorage'>"
	"      <arg type='h' direction='out' />"
	"    </method>"
	"  </interface>"
	"</node>";

G_DEFINE_TYPE (TrackerFilesInterface, tracker_files_interface, G_TYPE_OBJECT)

static void
tracker_files_interface_init (TrackerFilesInterface *files_interface)
{
	files_interface->fd = -1;
}

static void
handle_method_call (GDBusConnection       *connection,
                    const gchar           *sender,
                    const gchar           *object_path,
                    const gchar           *interface_name,
                    const gchar           *method_name,
                    GVariant              *parameters,
                    GDBusMethodInvocation *invocation,
                    gpointer               user_data)
{
	TrackerFilesInterface *files_interface = user_data;

	if (g_strcmp0 (method_name, "GetPersistenceStorage") == 0) {
		GVariant *out_parameters;
		g_autoptr (GUnixFDList) fd_list = NULL;
		g_autoptr (GError) error = NULL;
		int idx;

		if (files_interface->fd < 0) {
			g_dbus_method_invocation_return_error (invocation,
			                                       G_IO_ERROR,
			                                       G_IO_ERROR_FAILED,
			                                       "Could not create memfd");
			return;
		}

		fd_list = g_unix_fd_list_new ();
		idx = g_unix_fd_list_append (fd_list, files_interface->fd, &error);

		if (error) {
			g_dbus_method_invocation_return_gerror (invocation, error);
		} else {
			out_parameters = g_variant_new ("(h)", idx);
			g_dbus_method_invocation_return_value_with_unix_fd_list (invocation,
			                                                         out_parameters,
			                                                         fd_list);
		}
	} else {
		g_dbus_method_invocation_return_error (invocation,
		                                       G_DBUS_ERROR,
		                                       G_DBUS_ERROR_UNKNOWN_METHOD,
		                                       "Unknown method %s",
		                                       method_name);
	}
}

static GVariant *
create_extractor_config_variant (TrackerFilesInterface *files_interface)
{
	GVariantBuilder builder;
	g_autoptr (GVariant) max_bytes = NULL;

	g_variant_builder_init (&builder, G_VARIANT_TYPE ("a{sv}"));
	max_bytes = g_settings_get_value (files_interface->settings, "max-bytes");
	g_variant_builder_add (&builder, "{sv}", "max-bytes", max_bytes);

	if (files_interface->priority_graphs)
		g_variant_builder_add (&builder, "{sv}", "priority-graphs", files_interface->priority_graphs);

#ifdef HAVE_POWER
	if (files_interface->power) {
		g_variant_builder_add (&builder, "{sv}", "on-battery",
		                       g_variant_new_boolean (tracker_power_get_on_battery (files_interface->power)));
		g_variant_builder_add (&builder, "{sv}", "on-low-battery",
		                       g_variant_new_boolean (tracker_power_get_on_low_battery (files_interface->power)));
	}
#endif

	return g_variant_builder_end (&builder);
}

static GVariant *
handle_get_property (GDBusConnection  *connection,
                     const gchar      *sender,
                     const gchar      *object_path,
                     const gchar      *interface_name,
                     const gchar      *property_name,
                     GError          **error,
                     gpointer          user_data)
{
	TrackerFilesInterface *files_interface = user_data;

	if (g_strcmp0 (object_path, "/org/freedesktop/Tracker3/Files") != 0 ||
	    g_strcmp0 (interface_name, "org.freedesktop.Tracker3.Files") != 0) {
		g_set_error (error, G_IO_ERROR, G_IO_ERROR_INVALID_DATA,
		             "Wrong object/interface");
		return NULL;
	}

	if (g_strcmp0 (property_name, "ExtractorConfig") == 0) {
		return create_extractor_config_variant (files_interface);
	} else {
		g_set_error (error, G_IO_ERROR, G_IO_ERROR_INVALID_DATA,
		             "Unknown property");
		return NULL;
	}
}

static void
tracker_files_interface_emit_changed (TrackerFilesInterface *files_interface)
{
	GVariantBuilder builder;
	GVariant *properties;

	g_variant_builder_init (&builder, G_VARIANT_TYPE ("a{sv}"));
	g_variant_builder_add (&builder, "{sv}", "ExtractorConfig",
	                       create_extractor_config_variant (files_interface));
	properties = g_variant_builder_end (&builder);

	g_dbus_connection_emit_signal (files_interface->connection,
	                               NULL,
	                               "/org/freedesktop/Tracker3/Files",
	                               "org.freedesktop.DBus.Properties",
	                               "PropertiesChanged",
	                               g_variant_new ("(s@a{sv}@as)",
	                                              "org.freedesktop.Tracker3.Files",
	                                              properties,
	                                              g_variant_new_array (G_VARIANT_TYPE ("s"), NULL, 0)),
	                               NULL);
}

static void
tracker_files_interface_constructed (GObject *object)
{
	TrackerFilesInterface *files_interface = TRACKER_FILES_INTERFACE (object);
	GDBusInterfaceVTable vtable = { handle_method_call, handle_get_property, NULL };
	g_autoptr (GDBusNodeInfo) introspection_data = NULL;

	G_OBJECT_CLASS (tracker_files_interface_parent_class)->constructed (object);

	introspection_data = g_dbus_node_info_new_for_xml (introspection_xml, NULL);
	files_interface->object_id =
		g_dbus_connection_register_object (files_interface->connection,
		                                   "/org/freedesktop/Tracker3/Files",
		                                   introspection_data->interfaces[0],
		                                   &vtable, object, NULL, NULL);

	files_interface->settings = g_settings_new ("org.freedesktop.Tracker3.Extract");
	g_signal_connect_swapped (files_interface->settings, "changed::max-bytes",
	                          G_CALLBACK (tracker_files_interface_emit_changed), object);

#ifdef HAVE_POWER
	files_interface->power = tracker_power_new ();
	if (files_interface->power) {
		g_signal_connect_swapped (files_interface->power, "notify::on-battery",
		                          G_CALLBACK (tracker_files_interface_emit_changed), object);
		g_signal_connect_swapped (files_interface->power, "notify::on-low-battery",
		                          G_CALLBACK (tracker_files_interface_emit_changed), object);
	}
#endif

	if (files_interface->fd < 0) {
#ifdef HAVE_MEMFD_CREATE
		files_interface->fd = memfd_create ("extract-persistent-storage",
		                                    MFD_CLOEXEC);
#else
		g_autofree gchar *path = NULL;

		path = g_strdup_printf ("%s/tracker-persistence.XXXXXX",
		                        g_get_tmp_dir ());
		files_interface->fd = g_mkstemp_full (path, 0, 0600);
		unlink (path);
#endif
	}
}

static void
tracker_files_interface_finalize (GObject *object)
{
	TrackerFilesInterface *files_interface = TRACKER_FILES_INTERFACE (object);

	g_dbus_connection_unregister_object (files_interface->connection,
	                                     files_interface->object_id);
	g_clear_object (&files_interface->connection);
	g_clear_object (&files_interface->settings);
#ifdef HAVE_POWER
	g_clear_object (&files_interface->power);
#endif

	if (files_interface->fd)
		close (files_interface->fd);

	G_OBJECT_CLASS (tracker_files_interface_parent_class)->finalize (object);
}

static void
tracker_files_interface_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
	TrackerFilesInterface *files_interface = TRACKER_FILES_INTERFACE (object);

	switch (prop_id) {
	case PROP_CONNECTION:
		files_interface->connection = g_value_dup_object (value);
		break;
	case PROP_PERSISTENCE_FD:
		files_interface->fd = dup (g_value_get_int (value));
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
tracker_files_interface_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
	TrackerFilesInterface *files_interface = TRACKER_FILES_INTERFACE (object);

	switch (prop_id) {
	case PROP_CONNECTION:
		g_value_set_object (value, files_interface->connection);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
tracker_files_interface_class_init (TrackerFilesInterfaceClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->constructed = tracker_files_interface_constructed;
	object_class->finalize = tracker_files_interface_finalize;
	object_class->set_property = tracker_files_interface_set_property;
	object_class->get_property = tracker_files_interface_get_property;

	props[PROP_CONNECTION] =
		g_param_spec_object ("connection",
		                     NULL, NULL,
		                     G_TYPE_DBUS_CONNECTION,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_PERSISTENCE_FD] =
		g_param_spec_int ("persistence-fd",
		                  NULL, NULL,
		                  G_MININT, G_MAXINT, -1,
		                  G_PARAM_WRITABLE |
		                  G_PARAM_CONSTRUCT_ONLY |
		                  G_PARAM_STATIC_STRINGS);

	g_object_class_install_properties (object_class, N_PROPS, props);
}

TrackerFilesInterface *
tracker_files_interface_new (GDBusConnection *connection)
{
	return g_object_new (TRACKER_TYPE_FILES_INTERFACE,
	                     "connection", connection,
	                     NULL);
}

TrackerFilesInterface *
tracker_files_interface_new_with_fd (GDBusConnection *connection,
                                     int              fd)
{
	return g_object_new (TRACKER_TYPE_FILES_INTERFACE,
	                     "connection", connection,
	                     "persistence-fd", fd,
	                     NULL);
}

int
tracker_files_interface_dup_fd (TrackerFilesInterface *files_interface)
{
	return dup (files_interface->fd);
}

void
tracker_files_interface_set_priority_graphs (TrackerFilesInterface *files_interface,
                                             GVariant              *graphs)
{
	gboolean changed = FALSE;

	if (!!graphs != !!files_interface->priority_graphs)
		changed = TRUE;
	else if (graphs && files_interface->priority_graphs &&
	         !g_variant_equal (graphs, files_interface->priority_graphs))
		changed = TRUE;

	g_clear_pointer (&files_interface->priority_graphs, g_variant_unref);
	if (graphs)
		files_interface->priority_graphs = g_variant_ref_sink (graphs);

	if (changed)
		tracker_files_interface_emit_changed (files_interface);
}
