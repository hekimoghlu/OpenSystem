/*
 * Copyright (C) 2014 - Collabora Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "config-miners.h"

#include "tracker-extract-controller.h"

#include "tracker-main.h"

#include <gio/gunixfdlist.h>

enum {
	PROP_0,
	PROP_DECORATOR,
	PROP_EXTRACTOR,
	PROP_CONNECTION,
	PROP_PERSISTENCE,
	N_PROPS,
};

static GParamSpec *props[N_PROPS] = { 0, };

struct _TrackerExtractController {
	GObject parent_instance;

	TrackerDecorator *decorator;
	TrackerExtract *extractor;
	TrackerExtractPersistence *persistence;
	GCancellable *cancellable;
	GDBusConnection *connection;
	GDBusProxy *index_proxy;
	GDBusProxy *miner_proxy;
	guint object_id;
	gboolean paused;
};

#define OBJECT_PATH "/org/freedesktop/Tracker3/Extract"

static const gchar *introspection_xml =
	"<node>"
	"  <interface name='org.freedesktop.Tracker3.Extract'>"
	"    <method name='Check' />"
	"    <signal name='Error'>"
	"      <arg type='a{sv}' name='data' direction='out' />"
	"    </signal>"
	"    <signal name='Progress'>"
	"      <arg type='s' name='status' />"
	"      <arg type='d' name='progress' />"
	"      <arg type='i' name='remaining_time' />"
	"    </signal>"
	"  </interface>"
	"</node>";

static void tracker_extract_controller_initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (TrackerExtractController,
                         tracker_extract_controller,
                         G_TYPE_OBJECT,
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE,
                                                tracker_extract_controller_initable_iface_init))

static void
update_paused_state (TrackerExtractController *controller,
                     gboolean                  pause)
{
	if (!!pause == !!controller->paused)
		return;

	if (pause)
		tracker_miner_pause (TRACKER_MINER (controller->decorator));
	else
		tracker_miner_resume (TRACKER_MINER (controller->decorator));

	controller->paused = pause;
}

static void
update_extract_config (TrackerExtractController *controller,
                       GDBusProxy               *proxy)
{
	GVariantIter iter;
	g_autoptr (GVariant) v = NULL;
	GVariant *value;
	gchar *key;

	v = g_dbus_proxy_get_cached_property (proxy, "ExtractorConfig");
	if (!v)
		return;

	g_variant_iter_init (&iter, v);

	while (g_variant_iter_next (&iter, "{sv}", &key, &value)) {
		if (g_strcmp0 (key, "max-bytes") == 0 &&
		    g_variant_is_of_type (value, G_VARIANT_TYPE_INT32)) {
			tracker_extract_set_max_text (controller->extractor,
			                              g_variant_get_int32 (value));
		} else if (g_strcmp0 (key, "on-battery") == 0 &&
		           g_variant_is_of_type (value, G_VARIANT_TYPE_BOOLEAN)) {
			tracker_decorator_set_throttled (controller->decorator,
			                                 g_variant_get_boolean (value));
		} else if (g_strcmp0 (key, "on-low-battery") == 0 &&
		           g_variant_is_of_type (value, G_VARIANT_TYPE_BOOLEAN)) {
			update_paused_state (controller, g_variant_get_boolean (value));
		} else if (g_strcmp0 (key, "priority-graphs") == 0 &&
		           g_variant_is_of_type (value, G_VARIANT_TYPE_STRING_ARRAY)) {
			const gchar **graphs = NULL;

			graphs = g_variant_get_strv (value, NULL);
			tracker_decorator_set_priority_graphs (controller->decorator, graphs);
			g_free (graphs);
		}

		g_free (key);
		g_variant_unref (value);
	}
}

static void
miner_properties_changed_cb (GDBusProxy *proxy,
                             GVariant   *changed_properties,
                             GStrv       invalidated_properties,
                             gpointer    user_data)
{
	update_extract_config (user_data, proxy);
}

static gboolean
set_up_persistence (TrackerExtractController  *controller,
                    GCancellable              *cancellable,
                    GError                   **error)
{
	g_autoptr (GUnixFDList) out_fd_list = NULL;
	g_autoptr (GVariant) variant = NULL;
	int idx, fd;

	variant = g_dbus_proxy_call_with_unix_fd_list_sync (controller->miner_proxy,
	                                                    "GetPersistenceStorage",
	                                                    NULL,
	                                                    G_DBUS_CALL_FLAGS_NO_AUTO_START,
	                                                    -1,
	                                                    NULL,
	                                                    &out_fd_list,
	                                                    cancellable,
	                                                    error);
	if (!variant)
		return FALSE;

	g_variant_get (variant, "(h)", &idx);
	fd = g_unix_fd_list_get (out_fd_list, idx, error);
	if (fd < 0)
		return FALSE;

	tracker_extract_persistence_set_fd (controller->persistence, fd);
	return TRUE;
}

static void
interface_method_call (GDBusConnection       *connection,
                       const gchar           *sender,
                       const gchar           *object_path,
                       const gchar           *interface_name,
                       const gchar           *method_name,
                       GVariant              *parameters,
                       GDBusMethodInvocation *invocation,
                       gpointer               user_data)
{
	TrackerExtractController *controller = user_data;

	if (g_strcmp0 (method_name, "Check") == 0) {
		tracker_decorator_check_unextracted (controller->decorator);
		g_dbus_method_invocation_return_value (invocation, NULL);
	} else {
		g_dbus_method_invocation_return_error (invocation,
		                                       G_DBUS_ERROR,
		                                       G_DBUS_ERROR_UNKNOWN_METHOD,
		                                       "Unknown method %s",
		                                       method_name);
	}
}

static gboolean
tracker_extract_controller_initable_init (GInitable     *initable,
                                          GCancellable  *cancellable,
                                          GError       **error)
{
	TrackerExtractController *controller;
	g_autoptr (GDBusNodeInfo) introspection_data = NULL;
	GDBusInterfaceVTable interface_vtable = {
		interface_method_call, NULL, NULL
	};

	controller = TRACKER_EXTRACT_CONTROLLER (initable);

	controller->miner_proxy = g_dbus_proxy_new_sync (controller->connection,
	                                                 G_DBUS_PROXY_FLAGS_DO_NOT_AUTO_START,
	                                                 NULL,
	                                                 NULL,
	                                                 "/org/freedesktop/Tracker3/Files",
	                                                 "org.freedesktop.Tracker3.Files",
	                                                 NULL,
	                                                 error);
	if (!controller->miner_proxy)
		return FALSE;

	g_signal_connect (controller->miner_proxy, "g-properties-changed",
	                  G_CALLBACK (miner_properties_changed_cb), controller);
	update_extract_config (controller, controller->miner_proxy);

	if (!set_up_persistence (controller, cancellable, error))
		return FALSE;

	introspection_data = g_dbus_node_info_new_for_xml (introspection_xml, error);
	if (!introspection_data)
		return FALSE;

	controller->object_id =
		g_dbus_connection_register_object (controller->connection,
						   OBJECT_PATH,
		                                   introspection_data->interfaces[0],
		                                   &interface_vtable,
		                                   initable,
		                                   NULL, error);
	if (controller->object_id == 0)
		return FALSE;

	return TRUE;
}

static void
tracker_extract_controller_initable_iface_init (GInitableIface *iface)
{
	iface->init = tracker_extract_controller_initable_init;
}

static void
decorator_raise_error_cb (TrackerDecorator         *decorator,
                          GFile                    *file,
                          gchar                    *msg,
                          gchar                    *extra,
                          TrackerExtractController *controller)
{
	g_autoptr (GError) error = NULL;
	g_autofree gchar *uri = NULL;
	GVariantBuilder builder;

	uri = g_file_get_uri (file);

	g_variant_builder_init (&builder, G_VARIANT_TYPE ("a{sv}"));
	g_variant_builder_add (&builder, "{sv}", "uri",
	                       g_variant_new_string (uri));
	g_variant_builder_add (&builder, "{sv}", "message",
	                       g_variant_new_string (msg));

	if (extra) {
		g_variant_builder_add (&builder, "{sv}", "extra-info",
		                       g_variant_new_string (extra));
	}

	g_dbus_connection_emit_signal (controller->connection,
	                               NULL,
	                               OBJECT_PATH,
	                               "org.freedesktop.Tracker3.Extract",
	                               "Error",
	                               g_variant_new ("(@a{sv})", g_variant_builder_end (&builder)),
	                               &error);

	if (error)
		g_warning ("Could not emit signal: %s\n", error->message);
}

static void
decorator_progress_cb (TrackerMiner             *miner,
                       const gchar              *status,
                       gdouble                   progress,
                       gint                      remaining_time,
                       TrackerExtractController *controller)
{
	g_autoptr (GError) error = NULL;

	g_dbus_connection_emit_signal (controller->connection,
	                               NULL,
	                               OBJECT_PATH,
	                               "org.freedesktop.Tracker3.Extract",
	                               "Progress",
	                               g_variant_new ("(sdi)", status, progress, remaining_time),
	                               &error);

	if (error)
		g_warning ("Could not emit signal: %s\n", error->message);
}

static void
tracker_extract_controller_constructed (GObject *object)
{
	TrackerExtractController *self = (TrackerExtractController *) object;

	G_OBJECT_CLASS (tracker_extract_controller_parent_class)->constructed (object);

	g_assert (self->decorator != NULL);

	g_signal_connect (self->decorator, "raise-error",
	                  G_CALLBACK (decorator_raise_error_cb), object);
	g_signal_connect (self->decorator, "progress",
	                  G_CALLBACK (decorator_progress_cb), object);
}

static void
tracker_extract_controller_set_property (GObject      *object,
                                         guint         param_id,
                                         const GValue *value,
                                         GParamSpec   *pspec)
{
	TrackerExtractController *self = (TrackerExtractController *) object;

	switch (param_id) {
	case PROP_DECORATOR:
		self->decorator = g_value_dup_object (value);
		break;
	case PROP_EXTRACTOR:
		self->extractor = g_value_dup_object (value);
		break;
	case PROP_CONNECTION:
		self->connection = g_value_dup_object (value);
		break;
	case PROP_PERSISTENCE:
		self->persistence = g_value_dup_object (value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, param_id, pspec);
		break;
	}
}

static void
tracker_extract_controller_dispose (GObject *object)
{
	TrackerExtractController *self = (TrackerExtractController *) object;

	if (self->connection && self->object_id) {
		g_dbus_connection_unregister_object (self->connection, self->object_id);
		self->object_id = 0;
	}

	g_clear_object (&self->decorator);
	g_clear_object (&self->extractor);
	g_clear_object (&self->index_proxy);
	g_clear_object (&self->persistence);

	G_OBJECT_CLASS (tracker_extract_controller_parent_class)->dispose (object);
}

static void
tracker_extract_controller_class_init (TrackerExtractControllerClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->constructed = tracker_extract_controller_constructed;
	object_class->dispose = tracker_extract_controller_dispose;
	object_class->set_property = tracker_extract_controller_set_property;

	props[PROP_DECORATOR] =
		g_param_spec_object ("decorator", NULL, NULL,
		                     TRACKER_TYPE_DECORATOR,
		                     G_PARAM_STATIC_STRINGS |
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY);
	props[PROP_EXTRACTOR] =
		g_param_spec_object ("extractor", NULL, NULL,
		                     TRACKER_TYPE_EXTRACT,
		                     G_PARAM_STATIC_STRINGS |
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY);
	props[PROP_CONNECTION] =
		g_param_spec_object ("connection", NULL, NULL,
		                     G_TYPE_DBUS_CONNECTION,
		                     G_PARAM_STATIC_STRINGS |
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY);
	props[PROP_PERSISTENCE] =
		g_param_spec_object ("persistence", NULL, NULL,
		                     TRACKER_TYPE_EXTRACT_PERSISTENCE,
		                     G_PARAM_STATIC_STRINGS |
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY);

	g_object_class_install_properties (object_class, N_PROPS, props);
}

static void
tracker_extract_controller_init (TrackerExtractController *self)
{
}

TrackerExtractController *
tracker_extract_controller_new (TrackerDecorator           *decorator,
                                TrackerExtract             *extractor,
                                GDBusConnection            *connection,
                                TrackerExtractPersistence  *persistence,
                                GError                    **error)
{
	g_return_val_if_fail (TRACKER_IS_DECORATOR (decorator), NULL);

	return g_initable_new (TRACKER_TYPE_EXTRACT_CONTROLLER,
	                       NULL, error,
	                       "decorator", decorator,
	                       "extractor", extractor,
	                       "connection", connection,
	                       "persistence", persistence,
	                       NULL);
}
