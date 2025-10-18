/*
 * Copyright (C) 2016, Red Hat Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "config-miners.h"

#include "tracker-extract-watchdog.h"

#include "tracker-files-interface.h"
#include <tracker-common.h>

#include <sys/socket.h>

#define REMOTE_FD_NUMBER 3

enum {
	STATUS,
	LOST,
	N_SIGNALS
};

static guint signals[N_SIGNALS] = { 0, };

enum {
	PROP_0,
	PROP_SPARQL_CONN,
	PROP_INDEXING_TREE,
	N_PROPS,
};

static GParamSpec *props[N_PROPS] = { 0, };

struct _TrackerExtractWatchdog {
	GObject parent_class;
	TrackerSparqlConnection *sparql_conn;
	GSubprocessLauncher *launcher;
	GSubprocess *extract_process;
	GCancellable *cancellable;
	GDBusConnection *conn;
	TrackerEndpoint *endpoint;
	TrackerFilesInterface *files_interface;
	TrackerIndexingTree *indexing_tree;
	guint progress_signal_id;
	guint error_signal_id;
	int persistence_fd;
};

G_DEFINE_TYPE (TrackerExtractWatchdog, tracker_extract_watchdog, G_TYPE_OBJECT)

static void
on_extract_progress_cb (GDBusConnection *conn,
                        const gchar     *sender_name,
                        const gchar     *object_path,
                        const gchar     *interface_name,
                        const gchar     *signal_name,
                        GVariant        *parameters,
                        gpointer         user_data)
{
	TrackerExtractWatchdog *watchdog = user_data;
	const gchar *status;
	gdouble progress;
	gint32 remaining;

	g_variant_get (parameters, "(&sdi)",
	               &status, &progress, &remaining);
	g_signal_emit (watchdog, signals[STATUS], 0,
	               status, progress, (gint) remaining);
}

static void
on_extract_error_cb (GDBusConnection *conn,
                     const gchar     *sender_name,
                     const gchar     *object_path,
                     const gchar     *interface_name,
                     const gchar     *signal_name,
                     GVariant        *parameters,
                     gpointer         user_data)
{
	g_autoptr (GVariant) uri = NULL, message = NULL, extra = NULL, child = NULL;
	GVariantIter iter;
	GVariant *value;
	gchar *key;

	child = g_variant_get_child_value (parameters, 0);
	g_variant_iter_init (&iter, child);

	while (g_variant_iter_next (&iter, "{sv}", &key, &value)) {
		if (g_strcmp0 (key, "uri") == 0)
			uri = g_variant_ref_sink (value);
		else if (g_strcmp0 (key, "message") == 0)
			message = g_variant_ref_sink (value);
		else if (g_strcmp0 (key, "extra-info") == 0)
			extra = g_variant_ref_sink (value);

		g_variant_unref (value);
		g_free (key);
	}

	if (g_variant_is_of_type (uri, G_VARIANT_TYPE_STRING) &&
	    g_variant_is_of_type (message, G_VARIANT_TYPE_STRING) &&
	    (!extra || g_variant_is_of_type (extra, G_VARIANT_TYPE_STRING))) {
		g_autoptr (GFile) file = NULL;

		file = g_file_new_for_uri (g_variant_get_string (uri, NULL));
		tracker_error_report (file,
		                      g_variant_get_string (message, NULL),
		                      extra ? g_variant_get_string (extra, NULL) : NULL);
	}
}

static void
clear_process_state (TrackerExtractWatchdog *watchdog)
{
	if (watchdog->cancellable)
		g_cancellable_cancel (watchdog->cancellable);

	if (watchdog->conn && watchdog->progress_signal_id) {
		g_dbus_connection_signal_unsubscribe (watchdog->conn,
		                                      watchdog->progress_signal_id);
		watchdog->progress_signal_id = 0;
	}

	if (watchdog->conn && watchdog->error_signal_id) {
		g_dbus_connection_signal_unsubscribe (watchdog->conn,
		                                      watchdog->error_signal_id);
		watchdog->error_signal_id = 0;
	}

	g_clear_object (&watchdog->cancellable);
	g_clear_object (&watchdog->extract_process);
	g_clear_object (&watchdog->files_interface);
	g_clear_object (&watchdog->endpoint);
	g_clear_object (&watchdog->launcher);
	g_clear_object (&watchdog->conn);
}

static void
indexed_directory_removed_cb (TrackerIndexingTree    *tree,
                              GFile                  *file,
                              TrackerExtractWatchdog *watchdog)
{
	/* Terminate extractor process, so it can abandon activity early on
	 * pre-unmount.
	 */
	if (watchdog->extract_process) {
		g_subprocess_send_signal (watchdog->extract_process, SIGTERM);
		clear_process_state (watchdog);
	}
}

static void
tracker_extract_watchdog_constructed (GObject *object)
{
	TrackerExtractWatchdog *watchdog = TRACKER_EXTRACT_WATCHDOG (object);

	g_signal_connect (watchdog->indexing_tree, "directory-removed",
	                  G_CALLBACK (indexed_directory_removed_cb), watchdog);

	G_OBJECT_CLASS (tracker_extract_watchdog_parent_class)->constructed (object);
}

static void
tracker_extract_watchdog_finalize (GObject *object)
{
	TrackerExtractWatchdog *watchdog = TRACKER_EXTRACT_WATCHDOG (object);

	if (watchdog->extract_process)
		g_subprocess_send_signal (watchdog->extract_process, SIGTERM);

	clear_process_state (watchdog);

	if (watchdog->persistence_fd)
		close (watchdog->persistence_fd);

	g_signal_handlers_disconnect_by_func (watchdog->indexing_tree,
	                                      indexed_directory_removed_cb,
	                                      watchdog);

	g_clear_object (&watchdog->sparql_conn);
	g_clear_object (&watchdog->indexing_tree);

	G_OBJECT_CLASS (tracker_extract_watchdog_parent_class)->finalize (object);
}

static void
tracker_extract_watchdog_set_property (GObject      *object,
                                       guint         prop_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
	TrackerExtractWatchdog *watchdog = TRACKER_EXTRACT_WATCHDOG (object);

	switch (prop_id) {
	case PROP_SPARQL_CONN:
		watchdog->sparql_conn = g_value_dup_object (value);
		break;
	case PROP_INDEXING_TREE:
		watchdog->indexing_tree = g_value_dup_object (value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
tracker_extract_watchdog_class_init (TrackerExtractWatchdogClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->constructed = tracker_extract_watchdog_constructed;
	object_class->finalize = tracker_extract_watchdog_finalize;
	object_class->set_property = tracker_extract_watchdog_set_property;

	signals[STATUS] = g_signal_new ("status",
	                                G_OBJECT_CLASS_TYPE (object_class),
	                                G_SIGNAL_RUN_LAST,
	                                0, NULL, NULL, NULL,
	                                G_TYPE_NONE, 3,
	                                G_TYPE_STRING,
	                                G_TYPE_DOUBLE,
	                                G_TYPE_INT);
	signals[LOST] = g_signal_new ("lost",
	                              G_OBJECT_CLASS_TYPE (object_class),
	                              G_SIGNAL_RUN_LAST,
	                              0, NULL, NULL, NULL,
	                              G_TYPE_NONE, 0);

	props[PROP_SPARQL_CONN] =
		g_param_spec_object ("sparql-conn",
		                     NULL, NULL,
		                     TRACKER_TYPE_SPARQL_CONNECTION,
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_INDEXING_TREE] =
		g_param_spec_object ("indexing-tree", NULL, NULL,
		                     TRACKER_TYPE_INDEXING_TREE,
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);

	g_object_class_install_properties (object_class, N_PROPS, props);
}

static void
tracker_extract_watchdog_init (TrackerExtractWatchdog *watchdog)
{
	watchdog->cancellable = g_cancellable_new ();
	watchdog->persistence_fd = -1;
}

TrackerExtractWatchdog *
tracker_extract_watchdog_new (TrackerSparqlConnection *sparql_conn,
                              TrackerIndexingTree     *indexing_tree)
{
	return g_object_new (TRACKER_TYPE_EXTRACT_WATCHDOG,
	                     "sparql-conn", sparql_conn,
	                     "indexing-tree", indexing_tree,
	                     NULL);
}

static void
on_new_connection_cb (GObject      *object,
                      GAsyncResult *res,
                      gpointer      user_data)
{
	TrackerExtractWatchdog *watchdog = user_data;
	g_autoptr (GError) error = NULL;

	watchdog->conn = g_dbus_connection_new_finish (res, &error);
	if (!watchdog->conn) {
		if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
			g_warning ("Could not create peer-to-peer D-Bus connection: %s", error->message);
		return;
	}

	/* Create an endpoint for this peer-to-peer connection */
	watchdog->endpoint =
		TRACKER_ENDPOINT (tracker_endpoint_dbus_new (watchdog->sparql_conn,
		                                             watchdog->conn,
		                                             NULL, NULL,
		                                             &error));
	if (error) {
		g_warning ("Could not create endpoint for metadata extractor: %s", error->message);
		return;
	}

	/* Disallow access to further endpoints */
	tracker_endpoint_set_allowed_services (watchdog->endpoint,
	                                       (const gchar *[]) { NULL });

	watchdog->progress_signal_id =
		g_dbus_connection_signal_subscribe (watchdog->conn,
		                                    NULL,
		                                    "org.freedesktop.Tracker3.Extract",
		                                    "Progress",
		                                    "/org/freedesktop/Tracker3/Extract",
		                                    NULL,
		                                    G_DBUS_SIGNAL_FLAGS_NONE,
		                                    on_extract_progress_cb,
		                                    watchdog,
		                                    NULL);
	watchdog->error_signal_id =
		g_dbus_connection_signal_subscribe (watchdog->conn,
		                                    NULL,
		                                    "org.freedesktop.Tracker3.Extract",
		                                    "Error",
		                                    "/org/freedesktop/Tracker3/Extract",
		                                    NULL,
		                                    G_DBUS_SIGNAL_FLAGS_NONE,
		                                    on_extract_error_cb,
		                                    watchdog,
		                                    NULL);

	if (watchdog->persistence_fd >= 0) {
		watchdog->files_interface =
			tracker_files_interface_new_with_fd (watchdog->conn,
			                                     watchdog->persistence_fd);
	} else {
		watchdog->files_interface =
			tracker_files_interface_new (watchdog->conn);
		watchdog->persistence_fd =
			tracker_files_interface_dup_fd (watchdog->files_interface);
	}

	g_dbus_connection_start_message_processing (watchdog->conn);
}

static void
wait_check_async_cb (GObject      *object,
                     GAsyncResult *res,
                     gpointer      user_data)
{
	TrackerExtractWatchdog *watchdog = user_data;
	g_autoptr (GError) error = NULL;

	if (!g_subprocess_wait_check_finish (watchdog->extract_process,
	                                     res, &error)) {
		if (g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
			return;

		g_warning ("Extractor subprocess died unexpectedly: %s", error->message);
		g_signal_emit (watchdog, signals[LOST], 0);
	} else {
		g_signal_emit (watchdog, signals[STATUS], 0, "Idle", 1.0, 0);
	}

	clear_process_state (watchdog);
}

static GStrv
get_indexed_folders (TrackerExtractWatchdog *watchdog)
{
	GArray *array;
	g_autoptr (GList) roots = NULL;
	GList *l;

	array = g_array_new (TRUE, FALSE, sizeof (gchar*));
	roots = tracker_indexing_tree_list_roots (watchdog->indexing_tree);

	for (l = roots; l; l = l->next) {
		GFile *file = l->data;
		gchar *path = NULL;

		path = g_file_get_path (file);
		if (path)
			g_array_append_val (array, path);
	}

	return (GStrv) g_array_free (array, FALSE);
}

static void
extractor_child_setup (gpointer user_data)
{
#ifdef HAVE_LANDLOCK
	const gchar * const *indexed_folders = user_data;

	if (!tracker_landlock_init (indexed_folders)) {
		g_critical ("Refusing to extract file data since Landlock could not be enabled. "
		            "Update your kernel to fix this warning.");
		_exit (0);
	}
#endif
}

static gboolean
setup_context (TrackerExtractWatchdog  *watchdog,
               GError                 **error)
{
	g_autoptr (GSocket) socket = NULL;
	g_autoptr (GIOStream) stream = NULL;
	g_autofree gchar *guid = NULL;
	int fd_pair[2];

	clear_process_state (watchdog);

	watchdog->cancellable = g_cancellable_new ();

	if (socketpair (AF_LOCAL, SOCK_STREAM, 0, fd_pair)) {
		g_set_error (error,
		             G_IO_ERROR,
		             G_IO_ERROR_FAILED,
		             "socketpair failed: %m");
		return FALSE;
	}

	watchdog->launcher = g_subprocess_launcher_new (G_SUBPROCESS_FLAGS_NONE);
	g_subprocess_launcher_take_fd (watchdog->launcher, fd_pair[1], REMOTE_FD_NUMBER);
	g_subprocess_launcher_setenv (watchdog->launcher,
	                              "GVFS_REMOTE_VOLUME_MONITOR_IGNORE", "1",
	                              TRUE);

	g_subprocess_launcher_set_child_setup (watchdog->launcher,
	                                       extractor_child_setup,
	                                       get_indexed_folders (watchdog),
	                                       (GDestroyNotify) g_strfreev);

	socket = g_socket_new_from_fd (fd_pair[0], error);
	if (!socket) {
		close (fd_pair[0]);
		return FALSE;
	}

	stream = G_IO_STREAM (g_socket_connection_factory_create_connection (socket));

	guid = g_dbus_generate_guid ();

	g_dbus_connection_new (stream,
	                       guid,
	                       G_DBUS_CONNECTION_FLAGS_DELAY_MESSAGE_PROCESSING |
	                       G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_SERVER |
	                       G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_ALLOW_ANONYMOUS |
	                       G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_REQUIRE_SAME_USER,
	                       NULL, watchdog->cancellable,
	                       on_new_connection_cb, watchdog);

	return TRUE;
}

static void
on_check_finished (GObject      *object,
                   GAsyncResult *res,
                   gpointer      user_data)
{
	g_autoptr (GVariant) ignore = NULL;
	g_autoptr (GError) error = NULL;

	ignore = g_dbus_connection_call_finish (G_DBUS_CONNECTION (object),
	                                        res, &error);
	if (error)
		g_warning ("Could not ask extractor to update: %s", error->message);
}

void
tracker_extract_watchdog_ensure_started (TrackerExtractWatchdog *watchdog)
{
	g_autoptr (GError) error = NULL;
	g_autofree gchar *current_dir = NULL;
	const gchar *extract_path;

	if (watchdog->extract_process) {
		if (watchdog->conn) {
			g_dbus_connection_call (watchdog->conn,
			                        NULL,
			                        "/org/freedesktop/Tracker3/Extract",
			                        "org.freedesktop.Tracker3.Extract",
			                        "Check",
			                        NULL, NULL,
			                        G_DBUS_CALL_FLAGS_NONE,
			                        -1,
			                        NULL,
			                        on_check_finished,
			                        watchdog);
		}

		return;
	}

	if (!setup_context (watchdog, &error)) {
		g_critical ("Could not setup context to spawn metadata extractor: %s", error->message);
		return;
	}

	current_dir = g_get_current_dir ();

	if (g_strcmp0 (current_dir, BUILDROOT) == 0)
		extract_path = BUILD_EXTRACTDIR "/localsearch-extractor-3";
	else
		extract_path = LIBEXECDIR "/localsearch-extractor-3";

	watchdog->extract_process =
		g_subprocess_launcher_spawn (watchdog->launcher,
		                             &error,
		                             extract_path,
		                             "--socket-fd", G_STRINGIFY (REMOTE_FD_NUMBER),
		                             NULL);

	if (watchdog->extract_process) {
		g_subprocess_wait_check_async (watchdog->extract_process,
		                               watchdog->cancellable,
		                               wait_check_async_cb, watchdog);
	} else {
		g_warning ("Could not launch metadata extractor: %s", error->message);
	}
}
