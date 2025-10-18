/*
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
#include <stdio.h>

#include <glib.h>
#include <glib/gi18n.h>
#include <glib/gprintf.h>
#include <gio/gio.h>
#include <locale.h>
#include <tinysparql.h>

#include <tracker-common.h>

#include "tracker-process.h"
#include "tracker-color.h"
#include "tracker-control-proxy.h"

#define ASK_FILE_QUERY \
	"/org/freedesktop/LocalSearch/queries/ask-file.rq"
#define DELETE_FOLDER_QUERY \
	"/org/freedesktop/LocalSearch/queries/delete-folder-recursive.rq"

static gboolean filesystem = FALSE;
static gchar *filename = NULL;

static GOptionEntry entries[] = {
	{ "filesystem", 's', 0, G_OPTION_ARG_NONE, &filesystem,
	  N_("Remove filesystem indexer database"),
	  NULL },
	{ "file", 'f', 0, G_OPTION_ARG_FILENAME, &filename,
	  N_("Erase indexed information about a file, works recursively for directories"),
	  N_("FILE") },
	{ NULL }
};

static int
delete_info_recursively (GFile *file)
{
	g_autoptr (TrackerSparqlConnection) connection = NULL;
	g_autoptr (TrackerSparqlStatement) ask_stmt = NULL, delete_stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	g_autoptr (TrackerControlIndex) control_proxy = NULL;
	g_autofree char *uri = NULL;
	g_autoptr (GError) error = NULL;

	connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
	                                                NULL, NULL, &error);

	if (error)
		goto error;

	uri = g_file_get_uri (file);

	/* First, query whether the item exists */
	ask_stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                    ASK_FILE_QUERY,
	                                                                    NULL,
	                                                                    &error);

	if (ask_stmt) {
		tracker_sparql_statement_bind_string (ask_stmt, "url", uri);
		cursor = tracker_sparql_statement_execute (ask_stmt, NULL, &error);
	}

	if (!cursor ||
	    !tracker_sparql_cursor_next (cursor, NULL, &error) ||
	    !tracker_sparql_cursor_get_boolean (cursor, 0)) {
		if (error)
			goto error;

		return EXIT_SUCCESS;
	}

	/* Now, delete the element recursively */
	g_print ("%s\n", _("Deleting…"));

	delete_stmt =
		tracker_sparql_connection_load_statement_from_gresource (connection,
		                                                         DELETE_FOLDER_QUERY,
		                                                         NULL,
		                                                         &error);

	if (delete_stmt) {
		tracker_sparql_statement_bind_string (delete_stmt, "uri", uri);
		tracker_sparql_statement_update (delete_stmt, NULL, &error);
	}

	if (error)
		goto error;

	g_print ("%s\n", _("The indexed data for this file has been deleted "
	                   "and will be reindexed again."));

	/* Request reindexing of this data, it was previously in the store. */
	control_proxy =
		tracker_control_index_proxy_new_for_bus_sync (G_BUS_TYPE_SESSION,
		                                              G_DBUS_PROXY_FLAGS_DO_NOT_CONNECT_SIGNALS,
		                                              "org.freedesktop.LocalSearch3.Control",
		                                              "/org/freedesktop/Tracker3/Miner/Files/Index",
		                                              NULL, &error);

	if (!control_proxy)
		goto error;

	if (!tracker_control_index_call_index_location_sync (control_proxy, uri,
	                                                     (const char *[]) { "", NULL },
	                                                     (const char *[]) { "", NULL },
	                                                     NULL, &error))
		goto error;

	return EXIT_SUCCESS;

error:
	g_warning ("%s", error->message);
	return EXIT_FAILURE;
}

static void
delete_location (GFile *dir)
{
	g_autoptr (GFileEnumerator) enumerator = NULL;
	g_autoptr (GError) error = NULL;
	g_autoptr (GFileInfo) info = NULL;

	enumerator = g_file_enumerate_children (dir,
	                                        G_FILE_ATTRIBUTE_STANDARD_NAME,
	                                        G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
	                                        NULL, &error);
	if (error) {
		if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_NOT_FOUND)) {
			g_critical ("Location does not have a Tracker DB: %s",
			            error->message);
		}

		return;
	}

	while ((info = g_file_enumerator_next_file (enumerator, NULL, NULL)) != NULL) {
		g_autoptr (GFile) child = NULL;

		child = g_file_enumerator_get_child (enumerator, info);

		if (!g_file_delete (child, NULL, &error)) {
			g_critical ("Failed to delete '%s': %s",
			            g_file_info_get_name (info),
			            error->message);
		}
	}

	if (!g_file_delete (dir, NULL, &error)) {
		g_critical ("Failed to delete '%s': %s",
		            g_file_info_get_name (info),
		            error->message);
	}
}

static void
delete_path (const char *path)
{
	g_autoptr (GFile) file = NULL;

	file = g_file_new_for_path (path);
	delete_location (file);
}

int
tracker_reset (int          argc,
               const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree char *cache_dir = NULL, *errors_dir = NULL;

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	context = g_option_context_new (NULL);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_set_summary (context, _("Erase the indexed data"));

	argv[0] = "localsearch reset";

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		return EXIT_FAILURE;
	}

	if (filename) {
		g_autoptr (GFile) file = NULL;
		gint retval;

		file = g_file_new_for_commandline_arg (filename);
		retval = delete_info_recursively (file);
		return retval;
	}

	if (!filesystem && tracker_term_is_tty ()) {
		char response[100] = { 0, };

		g_print ("%s ", _("The LocalSearch indexed data is about to be deleted, proceed? [y/N]"));
		fgets (response, 100, stdin);
		response[strlen (response) - 1] = '\0';

		/* TRANSLATORS: this is our test for a [y|N] question in the command line.
		 * A partial or full match will be considered an affirmative answer,
		 * it is intentionally lowercase, so please keep it like this.
		 */
		if (!g_str_has_prefix (_("yes"), response))
			return EXIT_FAILURE;
	}

	/* Terminate and reset database */
	tracker_process_stop (SIGKILL);

	cache_dir = g_build_filename (g_get_user_cache_dir (), "tracker3", "files", "errors", NULL);
	delete_path (cache_dir);

	errors_dir = g_build_filename (g_get_user_cache_dir (), "tracker3", "files", NULL);
	delete_path (errors_dir);

	return EXIT_SUCCESS;
}
