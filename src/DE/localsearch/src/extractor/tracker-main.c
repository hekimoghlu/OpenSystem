/*
 * Copyright (C) 2006, Jamie McCracken <jamiemcc@gnome.org>
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
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

#define _XOPEN_SOURCE
#include <time.h>
#include <stdlib.h>
#include <locale.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

#include <glib-object.h>
#include <glib-unix.h>
#include <glib/gi18n.h>
#include <glib/gprintf.h>
#include <gio/gio.h>

#ifndef G_OS_WIN32
#include <sys/resource.h>
#endif

#include <tracker-common.h>

#include "tracker-main.h"
#include "tracker-decorator.h"
#include "tracker-extract.h"
#include "tracker-extract-controller.h"
#include "tracker-extract-persistence.h"

#define ABOUT	  \
	"Tracker " PACKAGE_VERSION "\n"

#define LICENSE	  \
	"This program is free software and comes without any warranty.\n" \
	"It is licensed under version 2 or later of the General Public " \
	"License which can be viewed at:\n" \
	"\n" \
	"  http://www.gnu.org/licenses/gpl.txt\n"

#define MINER_FS_NAME_SUFFIX "LocalSearch3"

static GMainLoop *main_loop;
static TrackerSparqlConnection *conn;

static gchar *filename;
static gchar *mime_type;
static gchar *output_format_name;
static gboolean version;
static guint shutdown_timeout_id = 0;
static int socket_fd;

static GOptionEntry entries[] = {
	{ "file", 'f', 0,
	  G_OPTION_ARG_FILENAME, &filename,
	  N_("File to extract metadata for"),
	  N_("FILE") },
	{ "mime", 't', 0,
	  G_OPTION_ARG_STRING, &mime_type,
	  N_("MIME type for file (if not provided, this will be guessed)"),
	  N_("MIME") },
	{ "output-format", 'o', 0, G_OPTION_ARG_STRING, &output_format_name,
	  N_("Output results format: “turtle”, “trig” or “json-ld”"),
	  N_("FORMAT") },
	{ "socket-fd", 's', 0,
	  G_OPTION_ARG_INT, &socket_fd,
	  N_("Socket file descriptor for peer-to-peer communication"),
	  N_("FD") },
	{ "version", 'V', 0,
	  G_OPTION_ARG_NONE, &version,
	  N_("Displays version information"),
	  NULL },
	{ NULL }
};

static void
initialize_priority_and_scheduling (void)
{
	/* Set CPU priority */
	tracker_sched_idle ();

	/* Set disk IO priority and scheduling */
	tracker_ioprio_init ();

	/* Set process priority:
	 * The nice() function uses attribute "warn_unused_result" and
	 * so complains if we do not check its returned value. But it
	 * seems that since glibc 2.2.4, nice() can return -1 on a
	 * successful call so we have to check value of errno too.
	 * Stupid...
	 */
	TRACKER_NOTE (CONFIG, g_message ("Setting priority nice level to 19"));

	if (nice (19) == -1) {
		const gchar *str = g_strerror (errno);

		TRACKER_NOTE (CONFIG, g_message ("Couldn't set nice value to 19, %s",
		                      str ? str : "no error given"));
	}
}

#ifndef HAVE_LIBSECCOMP
static gboolean
signal_handler (gpointer user_data)
{
	int signo = GPOINTER_TO_INT (user_data);

	static gboolean in_loop = FALSE;

	/* Die if we get re-entrant signals handler calls */
	if (in_loop) {
		_exit (EXIT_FAILURE);
	}

	switch (signo) {
	case SIGTERM:
	case SIGINT:
		in_loop = TRUE;
		g_main_loop_quit (main_loop);

		/* Fall through */
	default:
		if (g_strsignal (signo)) {
			g_debug ("Received signal:%d->'%s'",
			         signo,
			         g_strsignal (signo));
		}
		break;
	}

	return G_SOURCE_CONTINUE;
}

static void
initialize_signal_handler (void)
{
#ifndef G_OS_WIN32
	g_unix_signal_add (SIGTERM, signal_handler, GINT_TO_POINTER (SIGTERM));
	g_unix_signal_add (SIGINT, signal_handler, GINT_TO_POINTER (SIGINT));
#endif /* G_OS_WIN32 */
}
#endif /* !HAVE_LIBSECCOMP */

static int
run_standalone (void)
{
	g_autoptr (TrackerExtract) extract = NULL;
	g_autoptr (GFile) file = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree gchar *uri = NULL, *mime = NULL;
	TrackerResource *resource = NULL;
	GEnumClass *enum_class;
	GEnumValue *enum_value;
	TrackerRdfFormat output_format;
	TrackerExtractInfo *info;

	if (!output_format_name) {
		output_format_name = "turtle";
	}

	/* Look up the output format by name */
	enum_class = g_type_class_ref (TRACKER_TYPE_RDF_FORMAT);
	enum_value = g_enum_get_value_by_nick (enum_class, output_format_name);
	g_type_class_unref (enum_class);
	if (!enum_value) {
		g_printerr (N_("Unsupported serialization format “%s”\n"), output_format_name);
		return EXIT_FAILURE;
	}
	output_format = enum_value->value;

	tracker_locale_sanity_check ();

	file = g_file_new_for_commandline_arg (filename);

	if (mime_type) {
		mime = g_strdup (mime_type);
	} else {
		g_autoptr (GFileInfo) file_info = NULL;

		file_info = g_file_query_info (file,
		                               G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
		                               G_FILE_QUERY_INFO_NONE,
		                               NULL,
		                               &error);
		if (!file_info)
			goto error;

		mime = g_strdup (g_file_info_get_content_type (file_info));
	}

	uri = g_file_get_uri (file);

	extract = tracker_extract_new ();

	info = tracker_extract_file_sync (extract, uri, "_:content", mime, &error);
	if (!info)
		goto error;

	resource = tracker_extract_info_get_resource (info);

	if (resource) {
		if (output_format != TRACKER_RDF_FORMAT_JSON_LD) {
			TrackerNamespaceManager *namespaces;
			g_autofree char *turtle = NULL;

			/* If this was going into the tracker-store we'd generate a unique ID
			 * here, so that the data persisted across file renames.
			 */
			tracker_resource_set_identifier (resource, uri);

			G_GNUC_BEGIN_IGNORE_DEPRECATIONS
			namespaces = tracker_namespace_manager_get_default ();
			G_GNUC_END_IGNORE_DEPRECATIONS
			turtle = tracker_resource_print_rdf (resource, namespaces, output_format, NULL);

			g_print ("%s\n", turtle);
		} else {
			/* JSON-LD extraction */
			g_autofree char *json = NULL;

			/* If this was going into the tracker-store we'd generate a unique ID
			 * here, so that the data persisted across file renames.
			 */
			tracker_resource_set_identifier (resource, uri);

			/* We are using "deprecated" API here as the pretty printed output is
			 * nicer than with `tracker_resource_print_rdf()`, which uses the
			 * generic serializer.
			 */
			G_GNUC_BEGIN_IGNORE_DEPRECATIONS
			json = tracker_resource_print_jsonld (resource, NULL);
			G_GNUC_END_IGNORE_DEPRECATIONS

			g_print ("%s\n", json);
		}
	} else {
		g_printerr ("%s: %s\n",
		         uri,
		         _("No metadata or extractor modules found to handle this file"));
	}

	tracker_extract_info_unref (info);

	return EXIT_SUCCESS;
 error:
	g_printerr ("%s, %s\n",
	            _("Metadata extraction failed"),
	            error->message);

	return EXIT_FAILURE;
}

static void
on_decorator_items_available (TrackerDecorator *decorator)
{
	if (shutdown_timeout_id) {
		g_source_remove (shutdown_timeout_id);
		shutdown_timeout_id = 0;
	}
}

static gboolean
shutdown_timeout_cb (gpointer user_data)
{
	GMainLoop *loop = user_data;

	g_debug ("Shutting down after 10 seconds inactivity");
	g_main_loop_quit (loop);
	shutdown_timeout_id = 0;
	return G_SOURCE_REMOVE;
}

static void
on_decorator_finished (TrackerDecorator *decorator,
                       GMainLoop        *loop)
{
	if (shutdown_timeout_id != 0)
		return;

	shutdown_timeout_id = g_timeout_add_seconds (10, shutdown_timeout_cb,
	                                             main_loop);
}

TrackerSparqlConnection *
tracker_main_get_connection (void)
{
	return conn;
}

static int
do_main (int argc, char *argv[])
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (GError) error = NULL;
	g_autoptr (TrackerExtract) extract = NULL;
	g_autoptr (TrackerDecorator) decorator = NULL;
	g_autoptr (TrackerExtractController) controller = NULL;
	g_autoptr (GMainLoop) loop = NULL;
	g_autoptr (GDBusConnection) connection = NULL;
	g_autoptr (TrackerExtractPersistence) persistence = NULL;
	g_autoptr (TrackerSparqlConnection) sparql_connection = NULL;

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	/* Translators: this message will appear immediately after the  */
	/* usage string - Usage: COMMAND [OPTION]... <THIS_MESSAGE>     */
	context = g_option_context_new (_("— Extract file meta data"));

	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_parse (context, &argc, &argv, &error);

	if (error) {
		g_printerr ("%s\n", error->message);
		return EXIT_FAILURE;
	}

	if (!filename && mime_type) {
		g_autofree gchar *help = NULL;

		g_printerr ("%s\n\n",
		            _("Filename and mime type must be provided together"));

		help = g_option_context_get_help (context, TRUE, NULL);
		g_printerr ("%s", help);
		return EXIT_FAILURE;
	}

	if (version) {
		g_print ("\n" ABOUT "\n" LICENSE "\n");
		return EXIT_SUCCESS;
	}

	g_set_application_name ("tracker-extract");

	setlocale (LC_ALL, "");

	if (!tracker_extract_module_manager_init ())
		return EXIT_FAILURE;

	tracker_module_manager_load_modules ();

	/* Set conditions when we use stand alone settings */
	if (filename) {
		return run_standalone ();
	}

	if (socket_fd) {
		g_autoptr (GSocket) socket = NULL;
		g_autoptr (GIOStream) stream = NULL;

		socket = g_socket_new_from_fd (socket_fd, &error);

		if (socket) {
			stream = G_IO_STREAM (g_socket_connection_factory_create_connection (socket));
			connection = g_dbus_connection_new_sync (stream,
			                                         NULL,
			                                         G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT,
			                                         NULL, NULL, &error);
		}
	} else {
		g_warning ("The --socket-fd argument is mandatory");
		return EXIT_FAILURE;
	}

	if (!connection) {
		g_critical ("Could not create DBus connection: %s\n",
		            error->message);
		return EXIT_FAILURE;
	}

	extract = tracker_extract_new ();

	sparql_connection = conn =
		tracker_sparql_connection_bus_new (NULL,
		                                   NULL, connection,
		                                   &error);

	if (error) {
		g_critical ("Could not connect to filesystem miner endpoint: %s",
		            error->message);
		return EXIT_FAILURE;
	}

	persistence = tracker_extract_persistence_new ();

	decorator = tracker_decorator_new (sparql_connection, extract, persistence);

	tracker_locale_sanity_check ();

	controller = tracker_extract_controller_new (decorator,
	                                             extract,
	                                             connection,
	                                             persistence,
	                                             &error);
	if (error) {
		g_critical ("Could not create extraction controller: %s", error->message);
		return EXIT_FAILURE;
	}

	/* Main loop */
	loop = main_loop = g_main_loop_new (NULL, FALSE);

	g_signal_connect (decorator, "finished",
	                  G_CALLBACK (on_decorator_finished),
	                  main_loop);
	g_signal_connect (decorator, "items-available",
	                  G_CALLBACK (on_decorator_items_available),
	                  main_loop);

	tracker_miner_start (TRACKER_MINER (decorator));

#ifndef HAVE_LIBSECCOMP
	/* Play nice with coverage/valgrind/etc */
	initialize_signal_handler ();
#endif

	g_main_loop_run (main_loop);

	tracker_miner_stop (TRACKER_MINER (decorator));

	/* Shutdown subsystems */
	tracker_module_manager_shutdown_modules ();
	tracker_sparql_connection_close (sparql_connection);

	return EXIT_SUCCESS;
}

int
main (int argc, char *argv[])
{
	/* This function is untouchable! Add things to do_main() */

	/* This makes sure we don't steal all the system's resources */
	initialize_priority_and_scheduling ();

	if (!tracker_seccomp_init ())
		g_assert_not_reached ();

	return do_main (argc, argv);
}
