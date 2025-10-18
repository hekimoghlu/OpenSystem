/*
 * Copyright (C) 2025, Red Hat Inc.
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

#include "tracker-application.h"

#include "tracker-controller.h"
#include "tracker-miner-files.h"

#ifdef HAVE_MALLOC_TRIM
#include <malloc.h>
#endif

#include <glib/gi18n.h>

#define DBUS_NAME_SUFFIX "LocalSearch3"
#define LEGACY_DBUS_NAME_SUFFIX "Tracker3.Miner.Files"
#define DBUS_PATH "/org/freedesktop/Tracker3/Miner/Files"

#define ABOUT	  \
	"LocalSearch " PACKAGE_VERSION "\n"

#define LICENSE	  \
	"This program is free software and comes without any warranty.\n" \
	"It is licensed under version 2 or later of the General Public " \
	"License which can be viewed at:\n" \
	"\n" \
	"  http://www.gnu.org/licenses/gpl.txt\n"

#define CORRUPT_FILE_NAME ".localsearch.corrupted"
#define CONFIG_FILE ".config.gvariant"

static GOptionEntry entries[] = {
	{ "no-daemon", 'n', 0,
	  G_OPTION_ARG_NONE, NULL,
	  N_("Runs until all configured locations are indexed and then exits"),
	  NULL },
	{ "eligible", 'e', 0,
	  G_OPTION_ARG_FILENAME, NULL,
	  N_("Checks if FILE is eligible for being mined based on configuration"),
	  N_("FILE") },
	{ "dry-run", 'r', 0,
	  G_OPTION_ARG_NONE, NULL,
	  N_("Avoids changes in the filesystem"),
	  NULL },
	{ "version", 'V', 0,
	  G_OPTION_ARG_NONE, NULL,
	  N_("Displays version information"),
	  NULL },
	{ NULL }
};

typedef struct {
	TrackerSparqlConnection *sparql_conn;
	GDBusConnection *dbus_conn;
	GMainLoop *main_loop;
	GMutex mutex;
	GCond cond;
	GThread *thread;
	gboolean initialized;
	GError *error;
} EndpointThreadData;

static EndpointThreadData *endpoint_thread_data = NULL;

typedef struct _IndexerInstance IndexerInstance;
struct _IndexerInstance
{
	TrackerMiner *indexer;
	TrackerSparqlConnection *sparql_conn;
	TrackerIndexingTree *indexing_tree;
};

struct _TrackerApplication
{
	GApplication parent_instance;

	TrackerMiner *miner_files;
	TrackerMinerProxy *proxy;
	GDBusProxy *systemd_proxy;
	TrackerMonitor *monitor;
	TrackerStorage *storage;
	TrackerController *controller;
#if GLIB_CHECK_VERSION (2, 64, 0)
	GMemoryMonitor *memory_monitor;
#endif
	TrackerFilesInterface *files_interface;

	IndexerInstance main_instance;

	guint wait_settle_id;
	guint cleanup_id;
	guint domain_watch_id;
	guint legacy_name_id;
	guint no_daemon : 1;
	guint dry_run : 1;
	guint got_error : 1;
};

G_DEFINE_TYPE (TrackerApplication, tracker_application, G_TYPE_APPLICATION)

gpointer
endpoint_thread_func (gpointer user_data)
{
	EndpointThreadData *data = user_data;
	TrackerEndpointDBus *endpoint;
	GMainContext *main_context;

	main_context = g_main_context_new ();
	g_main_context_push_thread_default (main_context);

	endpoint = tracker_endpoint_dbus_new (data->sparql_conn,
	                                      data->dbus_conn,
	                                      NULL,
	                                      NULL,
	                                      &data->error);

	data->main_loop = g_main_loop_new (main_context, FALSE);

	g_mutex_lock (&data->mutex);
	data->initialized = TRUE;
	g_cond_signal (&data->cond);
	g_mutex_unlock (&data->mutex);

	if (!data->error)
		g_main_loop_run (data->main_loop);

	g_main_context_pop_thread_default (main_context);
	g_main_loop_unref (data->main_loop);
	g_main_context_unref (main_context);
	g_clear_object (&endpoint);

	return NULL;
}

gboolean
start_endpoint_thread (TrackerSparqlConnection  *conn,
                       GDBusConnection          *dbus_conn,
                       GError                  **error)
{
	g_assert (endpoint_thread_data == NULL);

	endpoint_thread_data = g_new0 (EndpointThreadData, 1);
	g_mutex_init (&endpoint_thread_data->mutex);
	g_cond_init (&endpoint_thread_data->cond);
	endpoint_thread_data->sparql_conn = conn;
	endpoint_thread_data->dbus_conn = dbus_conn;

	endpoint_thread_data->thread =
		g_thread_try_new ("SPARQL endpoint",
		                  endpoint_thread_func,
		                  endpoint_thread_data,
		                  error);

	if (!endpoint_thread_data->thread)
		return FALSE;

	g_mutex_lock (&endpoint_thread_data->mutex);

	while (!endpoint_thread_data->initialized) {
		g_cond_wait (&endpoint_thread_data->cond,
		             &endpoint_thread_data->mutex);
	}

	g_mutex_unlock (&endpoint_thread_data->mutex);

	if (endpoint_thread_data->error) {
		g_propagate_error (error, endpoint_thread_data->error);
		g_thread_join (endpoint_thread_data->thread);
		g_clear_pointer (&endpoint_thread_data, g_free);
		return FALSE;
	}

	return TRUE;
}

void
finish_endpoint_thread (void)
{
	if (!endpoint_thread_data)
		return;

	g_main_loop_quit (endpoint_thread_data->main_loop);

	g_thread_join (endpoint_thread_data->thread);

	g_clear_pointer (&endpoint_thread_data, g_free);
}

static GFile *
get_cache_dir (void)
{
	GFile *cache;

	if (MINER_FS_CACHE_LOCATION[0] == G_DIR_SEPARATOR) {
		cache = g_file_new_for_path (MINER_FS_CACHE_LOCATION);
	} else {
		g_autofree char *cache_dir = NULL;

		cache_dir = g_build_filename (g_get_user_cache_dir (),
		                              MINER_FS_CACHE_LOCATION,
		                              "files", NULL);
		cache = g_file_new_for_path (cache_dir);
	}

	return cache;
}

static TrackerSparqlConnection *
setup_connection (TrackerApplication  *app,
                  GFile               *store,
                  GError             **error)
{
	TrackerSparqlConnection *sparql_conn = NULL;
	g_autoptr (GFile) ontology = NULL, corrupt = NULL;
	g_autoptr (GError) internal_error = NULL;
	gboolean is_corrupted;
	TrackerSparqlConnectionFlags flags =
		TRACKER_SPARQL_CONNECTION_FLAGS_FTS_ENABLE_STEMMER |
		TRACKER_SPARQL_CONNECTION_FLAGS_FTS_ENABLE_UNACCENT;

	ontology = tracker_sparql_get_ontology_nepomuk ();

	corrupt = g_file_get_child (store, CORRUPT_FILE_NAME);
	is_corrupted = g_file_query_exists (corrupt, NULL);

	if (!store || !is_corrupted) {
		sparql_conn = tracker_sparql_connection_new (flags,
							     store,
							     ontology,
							     NULL,
							     &internal_error);
	}

	if (store) {
		if (is_corrupted ||
		    g_error_matches (internal_error, TRACKER_SPARQL_ERROR, TRACKER_SPARQL_ERROR_CORRUPT)) {
			g_autoptr (GFile) backup_location = NULL;
			g_autofree gchar *uri, *backup_uri = NULL;

			/* Move the database directory away, for possible forensics */
			uri = g_file_get_uri (store);
			backup_uri = g_strdup_printf ("%s.%" G_GINT64_FORMAT, uri, g_get_monotonic_time ());
			backup_location = g_file_new_for_uri (backup_uri);

			if (g_file_move (store, backup_location,
					 G_FILE_COPY_NONE,
					 NULL, NULL, NULL, NULL)) {
				g_autofree gchar *path = NULL;

				path = g_file_get_path (backup_location);
				g_message ("Database is corrupt, it is now backed up at %s. Reindexing from scratch", path);
				sparql_conn = tracker_sparql_connection_new (flags,
									     store,
									     ontology,
									     NULL,
									     &internal_error);
			}
		}
	}

	if (internal_error)
		g_propagate_error (error, g_steal_pointer (&internal_error));

	return sparql_conn;
}

static void
release_heap_memory (void)
{
#ifdef HAVE_MALLOC_TRIM
	malloc_trim (0);
#else
	g_debug ("release_heap_memory(): Doing nothing as malloc_trim() is not available on this platform.");
#endif
}

static gboolean
cleanup_cb (gpointer user_data)
{
	TrackerApplication *app = user_data;

	release_heap_memory ();

	app->cleanup_id = 0;

	return G_SOURCE_REMOVE;
}

static void
start_cleanup_timeout (TrackerApplication *app)
{
	if (app->cleanup_id == 0)
		app->cleanup_id = g_timeout_add_seconds (30, cleanup_cb, app);
}

static void
stop_cleanup_timeout (TrackerApplication *app)
{
	g_clear_handle_id (&app->cleanup_id, g_source_remove);
}

#if GLIB_CHECK_VERSION (2, 64, 0)
static void
on_low_memory (GMemoryMonitor            *monitor,
               GMemoryMonitorWarningLevel level,
               gpointer                   user_data)
{
	if (level > G_MEMORY_MONITOR_WARNING_LEVEL_LOW)
		release_heap_memory ();
}
#endif

static void
indexer_started_cb (TrackerApplication *app)
{
	stop_cleanup_timeout (app);
}

static void
indexer_finished_cb (TrackerApplication *app)
{
	if (app->no_daemon) {
		/* We're not sticking around for file updates, so stop
		 * the mainloop and exit.
		 */
		g_application_release (G_APPLICATION (app));
	}
}

static void
indexer_status_cb (TrackerApplication *app)
{
	g_autofree gchar *status = NULL;

	g_object_get (G_OBJECT (app->main_instance.indexer), "status", &status, NULL);
	if (g_strcmp0 (status, "Idle") == 0)
		start_cleanup_timeout (app);
	else
		stop_cleanup_timeout (app);
}

static void
indexer_corrupt_cb (TrackerApplication *app)
{
	g_autoptr (GFile) cache_dir = NULL, corrupt_file = NULL;
	g_autoptr (GError) error = NULL;

	cache_dir = get_cache_dir ();
	corrupt_file = g_file_get_child (cache_dir, CORRUPT_FILE_NAME);
	if (!g_file_set_contents (g_file_peek_path (corrupt_file), "", -1, &error))
		g_warning ("Could not mark database as corrupt: %s", error->message);

	g_warning ("Database corruption detected, bailing out");
	app->got_error = TRUE;
	g_application_quit (G_APPLICATION (app));
}

static void
start_indexer (TrackerApplication *app)
{
	if (!tracker_miner_is_started (app->main_instance.indexer)) {
		g_debug ("Starting filesystem miner...");
		tracker_miner_start (app->main_instance.indexer);
	}
}

static void
on_systemd_settled (TrackerApplication *app)
{
	g_debug ("Systemd session started");
	g_clear_handle_id (&app->wait_settle_id, g_source_remove);
	start_indexer (app);
}

static gboolean
wait_settle_cb (gpointer user_data)
{
	TrackerApplication *app = user_data;

	g_debug ("Waited 5 seconds for the system to settle");
	app->wait_settle_id = 0;
	start_indexer (app);

	return G_SOURCE_REMOVE;
}

static void
on_domain_vanished (GDBusConnection *connection,
                    const gchar     *name,
                    gpointer         user_data)
{
	TrackerApplication *app = user_data;

	g_message ("Domain %s vanished: quitting now.", name);
	g_application_quit (G_APPLICATION (app));
}

static void
shutdown_main_instance (TrackerApplication *app,
                        IndexerInstance    *instance)
{
	finish_endpoint_thread ();

	if (!app->dry_run &&
	    app->main_instance.indexing_tree) {
		g_autoptr (GFile) store, config;

		store = get_cache_dir ();
		config = g_file_get_child (store, CONFIG_FILE);

		tracker_indexing_tree_save_config (app->main_instance.indexing_tree,
		                                   config, NULL);
	}

	g_clear_object (&instance->indexer);
	g_clear_object (&instance->indexing_tree);
	g_clear_object (&instance->sparql_conn);
}

static gboolean
initialize_main_instance (TrackerApplication  *app,
                          IndexerInstance     *instance,
                          GDBusConnection     *dbus_conn,
                          GError             **error)
{
	g_autoptr (GFile) store = NULL;

	if (!app->dry_run) {
		store = get_cache_dir ();
		tracker_error_report_init (store);
	}

	instance->sparql_conn = setup_connection (app, store, error);
	if (!instance->sparql_conn)
		return FALSE;

	if (instance->sparql_conn) {
		instance->indexing_tree = tracker_indexing_tree_new ();
		instance->indexer = tracker_miner_files_new (instance->sparql_conn,
		                                             instance->indexing_tree,
		                                             app->monitor);
	}

	if (!start_endpoint_thread (instance->sparql_conn, dbus_conn, error))
		return FALSE;

	g_signal_connect_swapped (instance->indexer, "started",
	                          G_CALLBACK (indexer_started_cb),
	                          app);
	g_signal_connect_swapped (instance->indexer, "finished",
	                          G_CALLBACK (indexer_finished_cb),
	                          app);
	g_signal_connect_swapped (instance->indexer, "notify::status",
	                          G_CALLBACK (indexer_status_cb),
	                          app);
	g_signal_connect_swapped (instance->indexer, "corrupt",
	                          G_CALLBACK (indexer_corrupt_cb),
	                          app);

	return TRUE;
}

static void
tracker_application_finalize (GObject *object)
{
	TrackerApplication *app = TRACKER_APPLICATION (object);

	shutdown_main_instance (app, &app->main_instance);

	g_clear_handle_id (&app->domain_watch_id, g_bus_unwatch_name);

	g_clear_object (&app->files_interface);
	g_clear_object (&app->controller);
	g_clear_object (&app->proxy);
	g_clear_object (&app->storage);
	g_clear_object (&app->systemd_proxy);
	g_clear_object (&app->monitor);

	G_OBJECT_CLASS (tracker_application_parent_class)->finalize (object);
}

static gboolean
tracker_application_dbus_register (GApplication     *application,
				   GDBusConnection  *dbus_conn,
				   const char       *object_path,
				   GError          **error)
{
	TrackerApplication *app = TRACKER_APPLICATION (application);
	g_autofree char *legacy_dbus_name = NULL;
	gboolean wait_settle = FALSE;

	if (!initialize_main_instance (app, &app->main_instance, dbus_conn, error))
		return FALSE;

	if (!app->no_daemon &&
	    g_strcmp0 (DOMAIN_PREFIX, "org.freedesktop") != 0) {
		g_debug ("tracker-miner-fs-3 running for domain " DOMAIN_PREFIX ". "
		         "The service will exit when " DOMAIN_PREFIX " disappears from the bus.");
		app->domain_watch_id =
			g_bus_watch_name_on_connection (dbus_conn, DOMAIN_PREFIX,
			                                G_BUS_NAME_WATCHER_FLAGS_NONE,
			                                NULL, on_domain_vanished,
			                                app, NULL);
	}

	app->files_interface = tracker_files_interface_new (dbus_conn);

	app->controller = tracker_controller_new (app->main_instance.indexing_tree,
	                                          app->monitor, app->storage,
	                                          app->files_interface);

	app->proxy = tracker_miner_proxy_new (app->main_instance.indexer,
	                                      dbus_conn, DBUS_PATH, NULL, error);
	if (!app->proxy)
		return FALSE;

	if (!app->dry_run) {
		g_autoptr (GFile) store, config;

		store = get_cache_dir ();
		config = g_file_get_child (store, CONFIG_FILE);

		tracker_indexing_tree_check_config (app->main_instance.indexing_tree,
		                                    config);
	}

	/* Request legacy DBus name */
	legacy_dbus_name = g_strconcat (DOMAIN_PREFIX, ".", LEGACY_DBUS_NAME_SUFFIX, NULL);

	if (!tracker_dbus_request_name (dbus_conn, legacy_dbus_name, error))
		return FALSE;

	if (!tracker_term_is_tty ()) {
		app->systemd_proxy = g_dbus_proxy_new_for_bus_sync (TRACKER_IPC_BUS,
		                                                    G_DBUS_PROXY_FLAGS_NONE,
		                                                    NULL,
		                                                    "org.freedesktop.systemd1",
		                                                    "/org/freedesktop/systemd1",
		                                                    "org.freedesktop.systemd1.Manager",
		                                                    NULL, NULL);
		if (app->systemd_proxy) {
			const char *finished_states[] = { "running", "degraded", NULL };
			g_autoptr (GVariant) v = NULL;
			const char *state = NULL;

			g_signal_connect_swapped (app->systemd_proxy,
			                          "g-signal::StartupFinished",
			                          G_CALLBACK (on_systemd_settled),
			                          app);
			g_dbus_proxy_call_sync (app->systemd_proxy,
			                        "Subscribe",
			                        NULL,
			                        G_DBUS_CALL_FLAGS_NONE,
			                        -1,
			                        NULL, NULL);

			v = g_dbus_proxy_get_cached_property (app->systemd_proxy, "SystemState");

			if (v) {
				state = g_variant_get_string (v, NULL);
				wait_settle = !g_strv_contains (finished_states, state);
			}
		}
	}

	if (wait_settle) {
		g_debug ("Waiting for the system to settle");
		app->wait_settle_id = g_timeout_add_seconds (5, wait_settle_cb, app);
	} else {
		start_indexer (app);
	}

	return TRUE;
}

static void
tracker_application_dbus_unregister (GApplication    *app,
				     GDBusConnection *dbus_conn,
				     const char      *object_path)
{
}

static gint
check_eligible (const char *eligible)
{
	g_autoptr (TrackerIndexingTree) indexing_tree = NULL;
	g_autoptr (TrackerStorage) storage = NULL;
	g_autoptr (TrackerController) controller = NULL;
	g_autoptr (GFile) file = NULL;
	g_autoptr (GFileInfo) info = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree gchar *path = NULL;
	gboolean exists = TRUE;
	gboolean indexable;
	gboolean parents_indexable = TRUE;
	gboolean is_dir;

	indexing_tree = tracker_indexing_tree_new ();
	storage = tracker_storage_new ();

	controller = tracker_controller_new (indexing_tree, NULL, storage, NULL);

	/* Start check */
	file = g_file_new_for_commandline_arg (eligible);
	info = g_file_query_info (file,
	                          G_FILE_ATTRIBUTE_STANDARD_TYPE ","
	                          G_FILE_ATTRIBUTE_STANDARD_IS_HIDDEN,
	                          G_FILE_QUERY_INFO_NONE,
	                          NULL,
	                          &error);

	if (g_error_matches (error, G_IO_ERROR, G_IO_ERROR_NOT_FOUND))
		exists = FALSE;

	if (info) {
		is_dir = g_file_info_get_file_type (info) == G_FILE_TYPE_DIRECTORY;
	} else {
		/* Assume not a dir */
		is_dir = FALSE;
	}

	path = g_file_get_path (file);

	g_print (exists ?
	         _("Data object “%s” currently exists") :
	         _("Data object “%s” currently does not exist"),
	         path);

	g_print ("\n");

	indexable = tracker_indexing_tree_file_is_indexable (indexing_tree, file, info);

	if (indexable) {
		GFile *root, *parent;
		GList *files = NULL, *l;

		root = tracker_indexing_tree_get_root (indexing_tree, file, NULL, NULL);
		parent = file;

		/* Still, a parent folder might be filtered out, figure it out */
		while (parent && !g_file_equal (parent, root)) {
			parent = g_file_get_parent (parent);
			files = g_list_prepend (files, parent);
		}

		for (l = files; l; l = l->next) {
			g_autofree gchar *dir_path = NULL;

			dir_path = g_file_get_path (l->data);

			if (tracker_indexing_tree_file_matches_filter (indexing_tree, TRACKER_FILTER_DIRECTORY, l->data)) {
				g_print (_("Parent directory “%s” is NOT eligible to be indexed (based on filters)"),
				         dir_path);
				g_print ("\n");
				parents_indexable = FALSE;
			} else if (tracker_file_is_hidden (l->data) &&
			           tracker_indexing_tree_get_filter_hidden (indexing_tree)) {
				g_print (_("Parent directory “%s” is NOT eligible to be indexed (hidden file)"),
				         dir_path);
				g_print ("\n");
				parents_indexable = FALSE;
			} else {
				if (!tracker_indexing_tree_parent_is_indexable (indexing_tree, l->data)) {
					g_print (_("Parent directory “%s” is NOT eligible to be indexed (based on content filters)"),
					         dir_path);
					g_print ("\n");
					parents_indexable = FALSE;
				}
			}

			if (!parents_indexable)
				break;
		}

		g_list_free_full (files, g_object_unref);
	} else {
		if (is_dir &&
		    tracker_indexing_tree_file_matches_filter (indexing_tree, TRACKER_FILTER_DIRECTORY, file)) {
			g_print ("  %s\n", _("Directory is NOT eligible to be indexed (based on filters)"));

		} else if (!is_dir &&
		           tracker_indexing_tree_file_matches_filter (indexing_tree, TRACKER_FILTER_FILE, file)) {
			g_print ("  %s\n", _("File is NOT eligible to be indexed (based on filters)"));
		} else if (tracker_file_is_hidden (file) &&
		           tracker_indexing_tree_get_filter_hidden (indexing_tree)) {
			g_print ("  %s\n", _("File is NOT eligible to be indexed (hidden file)"));
		} else {
			g_print ("  %s\n", _("File is NOT eligible to be indexed (not an indexed folder)"));
		}
	}


	if (indexable && parents_indexable) {
		g_print ("  %s\n",
		         is_dir ?
		         _("Directory is eligible to be indexed") :
		         _("File is eligible to be indexed"));
	}

	return (indexable && parents_indexable) ? EXIT_SUCCESS : EXIT_FAILURE;
}

static gint
tracker_application_handle_local_options (GApplication *application,
                                          GVariantDict *options)
{
	TrackerApplication *app = TRACKER_APPLICATION (application);
	char *eligible_file = NULL;

	if (g_variant_dict_contains (options, "version")) {
		g_print ("\n" ABOUT "\n" LICENSE "\n");
		return EXIT_SUCCESS;
	}

	if (g_variant_dict_lookup (options, "eligible", "^&ay", &eligible_file))
		return check_eligible (eligible_file);

	if (g_variant_dict_contains (options, "no-daemon"))
		app->no_daemon = TRUE;
	if (g_variant_dict_contains (options, "dry-run"))
		app->dry_run = TRUE;

	return G_APPLICATION_CLASS (tracker_application_parent_class)->handle_local_options (application, options);
}

static void
tracker_application_startup (GApplication *application)
{
	g_application_hold (application);

	G_APPLICATION_CLASS (tracker_application_parent_class)->startup (application);
}

static void
tracker_application_class_init (TrackerApplicationClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);
	GApplicationClass *application_class = G_APPLICATION_CLASS (klass);

	object_class->finalize = tracker_application_finalize;

	application_class->dbus_register  = tracker_application_dbus_register;
	application_class->dbus_unregister = tracker_application_dbus_unregister;
	application_class->handle_local_options = tracker_application_handle_local_options;
	application_class->startup = tracker_application_startup;
}

static void
tracker_application_init (TrackerApplication *application)
{
	g_autoptr (GError) error = NULL;

	g_application_add_main_option_entries (G_APPLICATION (application), entries);

	application->storage = tracker_storage_new ();

	application->monitor = tracker_monitor_new (&error);

	if (!application->monitor) {
		g_warning ("Failed to initialize file monitoring: %s", error->message);
		g_clear_error (&error);
	}

#if GLIB_CHECK_VERSION (2, 64, 0)
	application->memory_monitor = g_memory_monitor_dup_default ();
	g_signal_connect (application->memory_monitor, "low-memory-warning",
	                  G_CALLBACK (on_low_memory), NULL);
#endif
}

GApplication *
tracker_application_new (void)
{
	g_autofree char *dbus_name = NULL;

	dbus_name = g_strconcat (DOMAIN_PREFIX, ".", DBUS_NAME_SUFFIX, NULL);

	return g_object_new (TRACKER_TYPE_APPLICATION,
	                     "application-id", dbus_name,
	                     "flags", G_APPLICATION_IS_SERVICE,
	                     NULL);
}

gboolean
tracker_application_exit_in_error (TrackerApplication *app)
{
	return app->got_error;
}
