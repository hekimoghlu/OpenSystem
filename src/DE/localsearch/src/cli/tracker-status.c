/*
 * Copyright (C) 2006, Jamie McCracken <jamiemcc@gnome.org>
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
 * Copyright (C) 2014, Lanedo <martyn@lanedo.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "config-miners.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>

#include <glib.h>
#include <glib-unix.h>
#include <glib/gi18n.h>
#include <tinysparql.h>

#include <tracker-common.h>

#include "tracker-indexer-proxy.h"
#include "tracker-term-utils.h"
#include "tracker-color.h"
#include "tracker-cli-utils.h"

#define GROUP "Report"
#define KEY_URI "Uri"
#define KEY_MESSAGE "Message"
#define KEY_SPARQL "Sparql"

#define GET_STATS_QUERY \
	"/org/freedesktop/LocalSearch/queries/get-class-stats.rq"
#define COUNT_FILES_QUERY \
	"/org/freedesktop/LocalSearch/queries/count-files.rq"
#define COUNT_FOLDERS_QUERY \
	"/org/freedesktop/LocalSearch/queries/count-folders.rq"

#define LINK_STR "[🡕]" /* NORTH EAST SANS-SERIF ARROW, in consistence with systemd */

#define INDETERMINATE_ROOM ((int) strlen ("100.0%") - 1)

static gboolean show_stat;
static gboolean follow;
static gboolean watch;
static gchar **terms;

static gboolean indexer_paused;
static int indeterminate_pos = 0;

static GOptionEntry entries[] = {
	{ "follow", 'f', 0, G_OPTION_ARG_NONE, &follow,
	  N_("Follow status changes as they happen"),
	  NULL
	},
	{ "stat", 'a', 0, G_OPTION_ARG_NONE, &show_stat,
	  N_("Show statistics for current index / data set"),
	  NULL
	},
	{ "watch", 'w', G_OPTION_FLAG_NONE, G_OPTION_ARG_NONE, &watch,
	  N_("Watch changes to the database in real time (e.g. resources or files being added)"),
	  NULL
	},
	{ G_OPTION_REMAINING, 0, 0,
	  G_OPTION_ARG_STRING_ARRAY, &terms,
	  N_("search terms"),
	  N_("EXPRESSION") },
	{ NULL }
};

static int show_errors (gchar    **terms,
                        gboolean   piped);

typedef struct {
	char *graph;
	char *type;
	char *type_expanded;
	gint64 count;
} ClassStat;

static void
clear_stat (gpointer data)
{
	ClassStat *stat = data;

	g_free (stat->graph);
	g_free (stat->type_expanded);
	g_free (stat->type);
}

static void
print_link (const gchar *url)
{
	g_print ("\x1B]8;;%s\a" LINK_STR "\x1B]8;;\a", url);
}

static int
status_stat (void)
{
	g_autoptr (TrackerSparqlConnection) connection = NULL;
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	const char *last_graph = NULL;
	TrackerNamespaceManager *namespaces;
	g_autoptr (GArray) stats = NULL;
	g_autoptr (GError) error = NULL;
	int longest_class = 0;
	guint i;

	connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
	                                                NULL, NULL, &error);

	if (!connection) {
		g_printerr ("%s: %s\n",
		            _("Could not connect to LocalSearch"),
		            error->message);
		return EXIT_FAILURE;
	}

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                GET_STATS_QUERY,
	                                                                NULL,
	                                                                &error);
	if (stmt)
		cursor = tracker_sparql_statement_execute (stmt, NULL, &error);

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not get LocalSearch statistics"),
		            error->message);
		return EXIT_FAILURE;
	}

	tracker_term_pipe_to_pager ();

	namespaces = tracker_sparql_connection_get_namespace_manager (connection);
	stats = g_array_new (FALSE, FALSE, sizeof (ClassStat));
	g_array_set_clear_func (stats, clear_stat);

	while (tracker_sparql_cursor_next (cursor, NULL, NULL)) {
		g_autofree char *compressed_rdf_type = NULL;
		const gchar *graph, *rdf_type;
		int64_t rdf_type_count;
		ClassStat entry;

		graph = tracker_sparql_cursor_get_string (cursor, 0, NULL);
		rdf_type = tracker_sparql_cursor_get_string (cursor, 1, NULL);
		rdf_type_count = tracker_sparql_cursor_get_integer (cursor, 2);

		if (terms) {
			gint i, n_terms;
			gboolean show_rdf_type = FALSE;

			n_terms = g_strv_length (terms);

			for (i = 0;
			     i < n_terms && !show_rdf_type;
			     i++) {
				show_rdf_type = g_str_match_string (terms[i], rdf_type, TRUE);
			}

			if (!show_rdf_type) {
				continue;
			}
		}

		entry.graph = tracker_namespace_manager_compress_uri (namespaces, graph);
		entry.type = tracker_namespace_manager_compress_uri (namespaces, rdf_type);
		entry.type_expanded = g_strdup (rdf_type);
		entry.count = rdf_type_count;

		longest_class = MAX (longest_class, strlen (entry.type));

		g_array_append_val (stats, entry);
	}

	for (i = 0; i < stats->len; i++) {
		ClassStat *stat;
		int padding;

		stat = &g_array_index (stats, ClassStat, i);

		if (g_strcmp0 (stat->graph, last_graph) != 0) {
			g_autofree char *compressed_graph = NULL;

			g_print (BOLD_BEGIN "%s: " BOLD_END "\n", stat->graph);
			last_graph = stat->graph;
		}

		padding = longest_class + 1 - strlen (stat->type);
		g_print ("%*c%s", padding, ' ', stat->type);
		print_link (stat->type_expanded);
		g_print (": %" G_GINT64_FORMAT "\n", stat->count);
	}

	tracker_term_pager_close ();

	return EXIT_SUCCESS;
}

static int
status_run (void)
{
	if (show_stat) {
		return status_stat ();
	}

	/* All known options have their own exit points */
	g_warn_if_reached ();

	return EXIT_FAILURE;
}

static int
get_file_and_folder_count (int *files,
                           int *folders)
{
	g_autoptr (TrackerSparqlConnection) connection = NULL;
	g_autoptr (GError) error = NULL;

	connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
	                                                NULL, NULL, &error);

	if (files) {
		*files = 0;
	}

	if (folders) {
		*folders = 0;
	}

	if (!connection) {
		g_printerr ("%s: %s\n",
		            _("Could not connect to LocalSearch"),
		            error->message);
		return EXIT_FAILURE;
	}

	if (files) {
		g_autoptr (TrackerSparqlStatement) stmt = NULL;
		g_autoptr (TrackerSparqlCursor) cursor = NULL;

		stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
		                                                                COUNT_FILES_QUERY,
		                                                                NULL,
		                                                                &error);

		if (stmt)
			cursor = tracker_sparql_statement_execute (stmt, NULL, &error);

		if (error || !tracker_sparql_cursor_next (cursor, NULL, &error)) {
			g_printerr ("%s, %s\n",
			            _("Could not get LocalSearch statistics"),
			            error->message);
			return EXIT_FAILURE;
		}

		*files = tracker_sparql_cursor_get_integer (cursor, 0);
	}

	if (folders) {
		g_autoptr (TrackerSparqlStatement) stmt = NULL;
		g_autoptr (TrackerSparqlCursor) cursor = NULL;

		stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
		                                                                COUNT_FOLDERS_QUERY,
		                                                                NULL,
		                                                                &error);

		if (stmt)
			cursor = tracker_sparql_statement_execute (stmt, NULL, &error);

		if (error || !tracker_sparql_cursor_next (cursor, NULL, NULL)) {
			g_printerr ("%s, %s\n",
			            _("Could not get LocalSearch statistics"),
			            error->message);
			return EXIT_FAILURE;
		}

		*folders = tracker_sparql_cursor_get_integer (cursor, 0);
	}

	return EXIT_SUCCESS;
}

static gboolean
are_miners_finished (gboolean *paused)
{
	g_autoptr (TrackerIndexerMiner) indexer_proxy = NULL;
	g_auto (GStrv) apps = NULL, reasons = NULL;
	gboolean is_paused;
	double progress;

	indexer_proxy =
		tracker_indexer_miner_proxy_new_for_bus_sync (G_BUS_TYPE_SESSION,
		                                              G_DBUS_PROXY_FLAGS_DO_NOT_AUTO_START |
		                                              G_DBUS_PROXY_FLAGS_DO_NOT_CONNECT_SIGNALS,
		                                              "org.freedesktop.LocalSearch3",
		                                              "/org/freedesktop/Tracker3/Miner/Files",
		                                              NULL, NULL);
	if (!indexer_proxy)
		return FALSE;

	if (!tracker_indexer_miner_call_get_pause_details_sync (indexer_proxy,
	                                                        &apps,
	                                                        &reasons,
	                                                        NULL, NULL))
		return FALSE;

	if (!tracker_indexer_miner_call_get_progress_sync (indexer_proxy,
	                                                   &progress,
	                                                   NULL, NULL))
		return FALSE;

	is_paused = apps && apps[0] && reasons && reasons[0];

	*paused = is_paused;

	return !is_paused && progress == 1.0;
}

static gint
print_errors (GList *keyfiles)
{
	gint cols, col_len[2];
	gchar *col_header1, *col_header2;
	GList *l;

	tracker_term_dimensions (&cols, NULL);
	col_len[0] = cols / 2;
	col_len[1] = cols / 2 - 1;

	col_header1 = tracker_term_ellipsize (_("Path"), col_len[0], TRACKER_ELLIPSIZE_END);
	col_header2 = tracker_term_ellipsize (_("Message"), col_len[1], TRACKER_ELLIPSIZE_END);

	g_print (BOLD_BEGIN "%-*s %-*s" BOLD_END "\n",
	         col_len[0], col_header1,
	         col_len[1], col_header2);
	g_free (col_header1);
	g_free (col_header2);

	for (l = keyfiles; l; l = l->next) {
		GKeyFile *keyfile = l->data;
		gchar *uri, *message, *path, *str1, *str2;
		g_autoptr(GFile) file = NULL;

		uri = g_key_file_get_string (keyfile, GROUP, KEY_URI, NULL);
		file = g_file_new_for_uri (uri);
		path = g_file_get_path (file);
		g_free (uri);

		if (!g_file_query_exists (file, NULL)) {
			tracker_error_report_delete (file);
			continue;
		}

		message = g_key_file_get_string (keyfile, GROUP, KEY_MESSAGE, NULL);

		str1 = tracker_term_ellipsize (path, col_len[0], TRACKER_ELLIPSIZE_START);
		str2 = tracker_term_ellipsize (message, col_len[1], TRACKER_ELLIPSIZE_END);

		g_print ("%-*s %-*s\n",
		         col_len[0], str1,
		         col_len[1], str2);
		g_free (path);
		g_free (message);
		g_free (str1);
		g_free (str2);
	}

	return EXIT_SUCCESS;
}

static int
get_no_args (void)
{
	gchar *str;
	gchar *data_dir;
	guint64 remaining_bytes;
	gdouble remaining;
	gint files, folders;
	GList *keyfiles;
	gboolean use_pager, paused;

	use_pager = tracker_term_pipe_to_pager ();

	/* How many files / folders do we have? */
	if (get_file_and_folder_count (&files, &folders) != 0) {
		return EXIT_FAILURE;
	}

	g_print (_("Currently indexed"));
	g_print (": ");
	g_print (g_dngettext (NULL,
	                      "%d file",
	                      "%d files",
	                      files),
	         files);
	g_print (", ");
	g_print (g_dngettext (NULL,
	                      "%d folder",
	                      "%d folders",
	                      folders),
	         folders);
	g_print ("\n");

	/* How much space is left? */
	data_dir = g_build_filename (g_get_user_cache_dir (), "tracker3", NULL);

	remaining_bytes = tracker_file_system_get_remaining_space (data_dir);
	str = g_format_size (remaining_bytes);

	remaining = tracker_file_system_get_remaining_space_percentage (data_dir);
	g_print ("%s: %s (%3.2lf%%)\n",
	         _("Remaining space on database partition"),
	         str,
	         remaining);
	g_free (str);
	g_free (data_dir);

	/* Are we finished indexing? */
	if (!are_miners_finished (&paused)) {
		g_print (BOLD_BEGIN "%s" BOLD_END "\n",
		         paused ?
		         _("Indexer is paused") :
		         _("Data is still being indexed"));
	} else {
		g_print ("%s\n", _("Indexer is idle"));
	}

	keyfiles = tracker_cli_get_error_keyfiles ();

	if (keyfiles) {
		g_print (g_dngettext (NULL,
		                      "%d recorded failure",
		                      "%d recorded failures",
		                      g_list_length (keyfiles)),
		         g_list_length (keyfiles));

		g_print (":\n\n");

		if (use_pager) {
			print_errors (keyfiles);
		} else {
			gchar *all[2] = { "", NULL };
			show_errors ((GStrv) all, TRUE);
		}

		g_list_free_full (keyfiles, (GDestroyNotify) g_key_file_unref);
	}

	tracker_term_pager_close ();

	return EXIT_SUCCESS;
}

static int
show_errors (gchar    **terms,
             gboolean   piped)
{
	GList *keyfiles, *l;
	GKeyFile *keyfile;
	guint i;
	gboolean found = FALSE;

	keyfiles = tracker_cli_get_error_keyfiles ();

	for (i = 0; terms[i] != NULL; i++) {
		for (l = keyfiles; l; l = l->next) {
			g_autoptr(GFile) file = NULL;
			g_autofree gchar *uri = NULL, *path = NULL;

			keyfile = l->data;
			uri = g_key_file_get_string (keyfile, GROUP, KEY_URI, NULL);
			file = g_file_new_for_uri (uri);

			if (!g_file_query_exists (file, NULL)) {
				tracker_error_report_delete (file);
				continue;
			}

			path = g_file_get_path (file);

			if (strstr (path, terms[i])) {
				gchar *sparql = g_key_file_get_string (keyfile, GROUP, KEY_SPARQL, NULL);
				gchar *message = g_key_file_get_string (keyfile, GROUP, KEY_MESSAGE, NULL);

				found = TRUE;
				g_print (!piped ?
				         BOLD_BEGIN "URI:" BOLD_END " %s\n" :
				         "URI: %s\n", uri);

				if (message) {
					g_print (!piped ?
					         BOLD_BEGIN "%s:" BOLD_END " %s\n" :
					         "%s: %s\n",
					         _("Message"), message);
				}

				if (sparql) {
					g_print (!piped ?
					         BOLD_BEGIN "SPARQL:" BOLD_END " %s\n" :
					         "SPARQL: %s\n",
					         sparql);
				}

				g_print ("\n");

				g_free (sparql);
				g_free (message);
			}
		}
	}

	if (!found) {
		g_print (!piped ?
		         BOLD_BEGIN "%s" BOLD_END "\n" :
		         "%s\n",
		         _("No reports found"));
	}

	return EXIT_SUCCESS;
}

static gboolean
signal_handler (gpointer user_data)
{
	GMainLoop *main_loop = user_data;

	g_main_loop_quit (main_loop);

	return G_SOURCE_CONTINUE;
}

static void
initialize_signal_handler (GMainLoop *main_loop)
{
	g_unix_signal_add (SIGTERM, signal_handler, main_loop);
	g_unix_signal_add (SIGINT, signal_handler, main_loop);
}

static void
notifier_events_cb (TrackerNotifier         *notifier,
		    const gchar             *service,
		    const gchar             *graph,
		    GPtrArray               *events,
		    TrackerSparqlConnection *conn)
{
	TrackerNamespaceManager *namespaces;
	gint i;

	namespaces = tracker_sparql_connection_get_namespace_manager (conn);

	for (i = 0; i < events->len; i++) {
		TrackerNotifierEvent *event;
		g_autofree char *compressed_graph = NULL;

		event = g_ptr_array_index (events, i);
		compressed_graph = tracker_namespace_manager_compress_uri (namespaces,
		                                                           graph);
		g_print ("%s (%s)\n",
		         tracker_notifier_event_get_urn (event),
		         compressed_graph);
	}
}

static void
maybe_reset_line (void)
{
	if (tracker_term_is_tty ()) {
		g_print ("\033[2K");
		g_print ("\r");
	}
}

static void
print_indexer_status (TrackerIndexerMiner *indexer_proxy)
{
	g_auto (GStrv) pause_apps = NULL, pause_reasons = NULL;
	g_autofree char *status = NULL;
	double progress;

	if (!tracker_indexer_miner_call_get_status_sync (indexer_proxy,
	                                                 &status,
	                                                 NULL, NULL))
		return;

	if (!tracker_indexer_miner_call_get_progress_sync (indexer_proxy,
	                                                   &progress,
	                                                   NULL, NULL))
		return;

	maybe_reset_line ();

	if (progress > 0.0) {
		g_print ("[%5.1f%%]", progress * 100);
	} else {
		g_print ("[");
		if (indeterminate_pos > 0)
			g_print ("%*c", indeterminate_pos, ' ');
		g_print ("=");
		if (INDETERMINATE_ROOM - indeterminate_pos > 0)
			g_print ("%*c", INDETERMINATE_ROOM - indeterminate_pos, ' ');
		g_print ("]");

		indeterminate_pos++;
		if (indeterminate_pos > INDETERMINATE_ROOM)
			indeterminate_pos = 0;
	}

	g_print (" %s", status);

	if (!tracker_term_is_tty ())
		g_print ("\n");
}

static void
indexer_progress_cb (TrackerIndexerMiner *indexer_proxy,
                     const char          *status,
                     double               progress,
                     int                  remaining_time)
{
	if (!indexer_paused)
		print_indexer_status (indexer_proxy);
}

static void
indexer_paused_cb (TrackerIndexerMiner *indexer_proxy)
{
	maybe_reset_line ();
	g_print ("%s", _("Indexer is paused"));
	indexer_paused = TRUE;
}

static void
indexer_resumed_cb (TrackerIndexerMiner *indexer_proxy)
{
	indexer_paused = FALSE;
}

static int
status_follow (void)
{
	g_autoptr (TrackerIndexerMiner) indexer_proxy = NULL;
	g_autoptr (GMainLoop) main_loop = NULL;

	indexer_proxy =
		tracker_indexer_miner_proxy_new_for_bus_sync (G_BUS_TYPE_SESSION,
		                                              G_DBUS_PROXY_FLAGS_DO_NOT_AUTO_START,
		                                              "org.freedesktop.LocalSearch3",
		                                              "/org/freedesktop/Tracker3/Miner/Files",
		                                              NULL, NULL);
	if (!indexer_proxy)
		return EXIT_FAILURE;

	print_indexer_status (indexer_proxy);

	g_signal_connect (indexer_proxy, "progress",
	                  G_CALLBACK (indexer_progress_cb), NULL);
	g_signal_connect (indexer_proxy, "paused",
	                  G_CALLBACK (indexer_paused_cb), NULL);
	g_signal_connect (indexer_proxy, "resumed",
	                  G_CALLBACK (indexer_resumed_cb), NULL);

	main_loop = g_main_loop_new (NULL, FALSE);
	initialize_signal_handler (main_loop);
	g_main_loop_run (main_loop);

	if (tracker_term_is_tty ()) {
		/* Print the status line a last time, papering over the ^C */
		print_indexer_status (indexer_proxy);
		g_print ("\n");
	}

	return EXIT_SUCCESS;
}

static int
status_watch (void)
{
	g_autoptr (GMainLoop) main_loop = NULL;
	g_autoptr (TrackerSparqlConnection) sparql_connection = NULL;
	g_autoptr (TrackerNotifier) notifier = NULL;
	g_autoptr (GError) error = NULL;

	sparql_connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
	                                                       NULL, NULL, &error);

	if (!sparql_connection) {
		g_critical ("%s, %s",
		            _("Could not get SPARQL connection"),
		            error->message);
		return EXIT_FAILURE;
	}

	notifier = tracker_sparql_connection_create_notifier (sparql_connection);
	g_signal_connect (notifier, "events",
	                  G_CALLBACK (notifier_events_cb), sparql_connection);

	g_print ("%s\n", _("Now listening to database updates"));
	g_print ("%s\n", _("Press Ctrl+C to stop"));

	main_loop = g_main_loop_new (NULL, FALSE);
	initialize_signal_handler (main_loop);
	g_main_loop_run (main_loop);

	/* Carriage return, so we paper over the ^C */
	g_print ("\r");

	return EXIT_SUCCESS;
}

static int
status_run_default (void)
{
	return get_no_args ();
}

int
tracker_status (int          argc,
                const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (GError) error = NULL;

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	context = g_option_context_new (NULL);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_set_summary (context, _("Provide status and statistics on the data indexed"));

	argv[0] = "localsearch status";

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		return EXIT_FAILURE;
	}

	if (show_stat) {
		return status_run ();
	} else if (follow) {
		return status_follow ();
	} else if (watch) {
		return status_watch ();
	}

	if (terms) {
		gint result;

		tracker_term_pipe_to_pager ();
		result = show_errors (terms, FALSE);
		tracker_term_pager_close ();

		return result;
	}

	return status_run_default ();
}
