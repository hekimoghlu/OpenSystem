/*
 * Copyright (C) 2006, Jamie McCracken <jamiemcc@gnome.org>
 * Copyright (C) 2008-2010, Nokia <ivan.frade@nokia.com>
 * Copyright (C) 2014, SoftAtHome <contact@softathome.com>
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
#include <glib/gi18n.h>
#include <gio/gunixoutputstream.h>
#include <tinysparql.h>

#include <tracker-common.h>

#include "tracker-cli-utils.h"
#include "tracker-color.h"

#define GET_INFORMATION_ELEMENT_QUERY \
	"/org/freedesktop/LocalSearch/queries/get-information-element.rq"

#define INFO_OPTIONS_ENABLED() \
	(filenames && g_strv_length (filenames) > 0);

#define GROUP "Report"
#define KEY_URI "Uri"
#define KEY_MESSAGE "Message"
#define KEY_SPARQL "Sparql"
#define ERROR_MESSAGE "Extraction failed for this file. Some metadata will be missing."

#define LINK_STR "[🡕]" /* NORTH EAST SANS-SERIF ARROW, in consistence with systemd */

static gboolean inside_build_tree = FALSE;

static gchar **filenames;
static gboolean full_namespaces;
static gboolean plain_text_content;
static char *output_format = NULL;
static gboolean eligible;

static gboolean output_is_tty;

static GOptionEntry entries[] = {
	{ "full-namespaces", 'f', 0, G_OPTION_ARG_NONE, &full_namespaces,
	  N_("Show full namespaces (i.e. don’t use nie:title, use full URLs)"),
	  NULL,
	},
	{ "plain-text-content", 'c', 0, G_OPTION_ARG_NONE, &plain_text_content,
	  N_("Show plain text content if available for resources"),
	  NULL,
	},
	{ "output-format", 'o', 0, G_OPTION_ARG_STRING, &output_format,
	  N_("Output results format: “turtle”, “trig” or “json-ld”"),
	  N_("FORMAT") },
	{ "eligible", 'e', 0, G_OPTION_ARG_NONE, &eligible,
	  N_("Checks if FILE is eligible for being indexed"),
	  NULL },
	{ G_OPTION_REMAINING, 0, 0, G_OPTION_ARG_FILENAME_ARRAY, &filenames,
	  N_("FILE"),
	  N_("FILE")},
	{ NULL }
};

static TrackerSparqlStatement *
describe_statement_for_urns (TrackerSparqlConnection *conn,
                             GList                   *urns)
{
	TrackerSparqlStatement *stmt;
	g_autoptr (GString) str = NULL;
	GList *l;

	str = g_string_new ("DESCRIBE");

	for (l = urns; l; l = l->next)
		g_string_append_printf (str, " <%s>", (gchar*) l->data);

	stmt = tracker_sparql_connection_query_statement (conn,
	                                                  str->str,
	                                                  NULL, NULL);

	return stmt;
}

static void
free_object_list (gpointer data)
{
	g_list_free_full (data, g_free);
}

static void
accumulate_value (GHashTable  *values,
                  const gchar *pred,
                  const gchar *object)
{
	GList *list;

	list = g_hash_table_lookup (values, pred);

	if (!g_list_find_custom (list, object, (GCompareFunc) g_strcmp0)) {
		g_hash_table_steal (values, pred);
		list = g_list_prepend (list, g_strdup (object));
		g_hash_table_insert (values, g_strdup (pred), list);
	}
}

static void
print_link (const gchar *url)
{
	g_print ("\x1B]8;;%s\a" LINK_STR "\x1B]8;;\a", url);
}

static void
print_object (const gchar *object,
              int          multiline_padding)
{
	if (strchr (object, '\n')) {
		const gchar *p = object;
		const gchar *end;
		gboolean first = TRUE;

		while (p && *p) {
			if (!first)
				g_print ("%*c", multiline_padding, ' ');

			end = strchr (p, '\n');
			first = FALSE;

			if (end) {
				end++;
				g_print ("%.*s", (int) (end - p), p);
				p = end;
			} else {
				g_print ("%s", p);
				break;
			}
		}
	} else {
		g_print ("%s", object);
	}
}

static void
print_plain_values (const gchar             *subject,
                    GHashTable              *values,
                    TrackerNamespaceManager *namespaces,
                    int                      axis_column)
{
	GHashTableIter iter;
	const gchar *pred;
	GList *objects, *l;
	int link_padding = 0;
	gboolean show_links = !full_namespaces && output_is_tty;

	g_hash_table_iter_init (&iter, values);

	if (output_is_tty)
		g_print (BOLD_BEGIN "%s" BOLD_END ":\n", subject);
	else
		g_print ("%s:\n", subject);

	if (!full_namespaces && show_links)
		link_padding = g_utf8_strlen (LINK_STR, -1);

	while (g_hash_table_iter_next (&iter, (gpointer*) &pred, (gpointer*) &objects)) {
		int len, padding;

		len = g_utf8_strlen (pred, -1);
		padding = axis_column - len;
		g_print ("%*c%s", padding, ' ', pred);

		if (show_links) {
			g_autofree gchar *expanded = NULL;
			expanded = tracker_namespace_manager_expand_uri (namespaces, pred);
			print_link (expanded);
		}

		g_print (": ");

		for (l = objects; l; l = l->next) {
			g_autofree gchar *str = NULL;

			if (!full_namespaces && g_str_has_prefix (l->data, "http"))
				str = tracker_namespace_manager_compress_uri (namespaces, l->data);

			if (!str)
				str = g_strdup (l->data);

			if (l != objects)
				g_print ("%*c", axis_column + link_padding + 2, ' ');

			print_object (str, axis_column + link_padding + 2);

			if (show_links && g_strcmp0 (str, l->data) != 0)
				print_link (l->data);

			g_print ("\n");
		}
	}

	g_print ("\n");
}

static void
print_plain_objects (GHashTable              *objects,
                     TrackerNamespaceManager *namespaces,
                     int                      axis_column)
{
	GHashTableIter iter;
	gpointer subject, values;

	g_hash_table_iter_init (&iter, objects);

	while (g_hash_table_iter_next (&iter, &subject, &values))
		print_plain_values (subject, values, namespaces, axis_column);
}

static void
print_plain (TrackerSparqlCursor     *cursor,
             TrackerNamespaceManager *namespaces)
{
	g_autoptr (GHashTable) objects = NULL;
	GHashTable *values = NULL;
	gchar *last_subject = NULL;
	int longest_pred = 0;

	objects = g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
	                                 (GDestroyNotify) g_hash_table_unref);

	while (tracker_sparql_cursor_next (cursor, NULL, NULL)) {
		const gchar *subject = tracker_sparql_cursor_get_string (cursor, 0, NULL);
		const gchar *pred = tracker_sparql_cursor_get_string (cursor, 1, NULL);
		const gchar *object = tracker_sparql_cursor_get_string (cursor, 2, NULL);

		if (g_strcmp0 (subject, last_subject) != 0) {
			last_subject = g_strdup (subject);
			values = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, free_object_list);
			g_hash_table_insert (objects, last_subject, values);
		}

		/* Don't display nie:plainTextContent */
		if (!plain_text_content &&
		    strcmp (pred, TRACKER_PREFIX_NIE "plainTextContent") == 0)
			continue;

		if (full_namespaces) {
			accumulate_value (values, pred, object);
			longest_pred = MAX (longest_pred, g_utf8_strlen (pred, -1));
		} else {
			g_autofree gchar *compressed = NULL;
			compressed = tracker_namespace_manager_compress_uri (namespaces, pred);
			accumulate_value (values, compressed, object);
			longest_pred = MAX (longest_pred, g_utf8_strlen (compressed, -1));
		}
	}

	print_plain_objects (objects, namespaces, longest_pred + 1);
}

static void
serialize_cb (GObject *object,
              GAsyncResult *res,
              gpointer      user_data)
{
	GMainLoop *main_loop = user_data;
	g_autoptr (GInputStream) istream = NULL;
	g_autoptr (GOutputStream) ostream = NULL;
	g_autoptr (GError) error = NULL;

	istream = tracker_sparql_statement_serialize_finish (TRACKER_SPARQL_STATEMENT (object),
	                                                     res, &error);
	if (istream) {
		ostream = g_unix_output_stream_new (STDOUT_FILENO, FALSE);
		g_output_stream_splice (ostream, istream, G_OUTPUT_STREAM_SPLICE_NONE, NULL, &error);
		g_output_stream_close (ostream, NULL, NULL);
	}

	if (error)
		g_printerr ("%s\n", error->message);

	g_main_loop_quit (main_loop);
}

static gboolean
output_eligible_status_for_file (gchar   *path,
                                 GError **error)
{
	g_autofree char *tracker_miner_fs_path = NULL;

	if (inside_build_tree) {
		/* Developer convenience - use uninstalled version if running from build tree */
		tracker_miner_fs_path = g_build_filename (BUILDROOT, "src", "indexer", "localsearch-3", NULL);
	} else {
		tracker_miner_fs_path = g_build_filename (LIBEXECDIR, "localsearch-3", NULL);
	}

	{
		char *argv[] = {tracker_miner_fs_path, "--eligible", path, NULL };

		return g_spawn_sync (NULL, argv, NULL, G_SPAWN_DEFAULT, NULL, NULL, NULL, NULL, NULL, error);
	}
}

static void
print_errors (GList       *keyfiles,
              const gchar *file_uri)
{
	GList *l;
	GKeyFile *keyfile;
	g_autoptr (GFile) file = NULL;

	file = g_file_new_for_uri (file_uri);

	for (l = keyfiles; l; l = l->next) {
		g_autofree char *uri = NULL;
		g_autoptr (GFile) error_file = NULL;

		keyfile = l->data;
		uri = g_key_file_get_string (keyfile, GROUP, KEY_URI, NULL);
		error_file = g_file_new_for_uri (uri);

		if (g_file_equal (file, error_file)) {
			g_autofree char *message = g_key_file_get_string (keyfile, GROUP, KEY_MESSAGE, NULL);
			g_autofree char *sparql = g_key_file_get_string (keyfile, GROUP, KEY_SPARQL, NULL);

			if (message)
				g_print (CRIT_BEGIN "%s\n%s: %s" CRIT_END "\n",
					 ERROR_MESSAGE,
					 _("Error message"),
					 message);
			if (sparql)
				g_print ("SPARQL: %s\n", sparql);
			g_print ("\n");
		}
	}
}


static int
info_run (void)
{
	g_autoptr (TrackerSparqlConnection) connection = NULL;
	g_autoptr (GError) error = NULL;
	GList *urns = NULL;
	gchar **p;

	tracker_term_pipe_to_pager ();

	connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
	                                                NULL, NULL, &error);

	if (!connection) {
		g_printerr ("%s: %s\n",
		            _("Could not connect to LocalSearch"),
		            error->message);
		return EXIT_FAILURE;
	}

	for (p = filenames; *p; p++) {
		g_autoptr (TrackerSparqlStatement) stmt = NULL;
		g_autoptr (TrackerSparqlCursor) cursor = NULL;
		g_autofree gchar *uri = NULL, *query = NULL, *uri_scheme = NULL;
		g_autoptr (GList) keyfiles = NULL;
		gboolean found = FALSE;

		uri_scheme = g_uri_parse_scheme (*p);

		/* support both, URIs and local file paths */
		if (uri_scheme) {
			uri = g_strdup (*p);
		} else {
			g_autoptr (GFile) file = NULL;

			file = g_file_new_for_commandline_arg (*p);
			uri = g_file_get_uri (file);
		}

		stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
		                                                                GET_INFORMATION_ELEMENT_QUERY,
		                                                                NULL,
		                                                                &error);
		if (stmt) {
			tracker_sparql_statement_bind_string (stmt, "uri", uri);
			cursor = tracker_sparql_statement_execute (stmt, NULL, &error);
		}

		if (cursor) {
			while (tracker_sparql_cursor_next (cursor, NULL, NULL)) {
				const gchar *str;

				str = tracker_sparql_cursor_get_string (cursor, 0, NULL);
				urns = g_list_prepend (urns, g_strdup (str));
				found = TRUE;
			}
		}

		if (error) {
			g_printerr ("  %s, %s\n",
			            _("Unable to retrieve data for URI"),
			            error->message);
			g_clear_error (&error);
		}

		if (!found) {
			if (output_format) {
				g_print ("# No metadata available for <%s>\n", uri);
			} else {
				g_print ("  %s\n",
				         _("No metadata available for that URI"));
				output_eligible_status_for_file (*p, &error);

				if (error) {
					g_printerr ("%s: %s\n",
					            _("Could not get eligible status: "),
					            error->message);
					g_clear_error (&error);
				}

				keyfiles = tracker_cli_get_error_keyfiles ();
				if (keyfiles)
					print_errors (keyfiles, uri);
			}
		}
	}

	if (!urns)
		goto out;

	if (output_format) {
		g_autoptr (TrackerSparqlStatement) stmt = NULL;
		g_autoptr (GMainLoop) main_loop = NULL;
		g_autoptr (GEnumClass) enum_class = NULL;
		GEnumValue *enum_value;
		TrackerRdfFormat rdf_format;

		/* Look up the output format by name */
		enum_class = g_type_class_ref (TRACKER_TYPE_RDF_FORMAT);
		enum_value = g_enum_get_value_by_nick (enum_class, output_format);
		if (!enum_value) {
			g_printerr (N_("Unsupported serialization format “%s”\n"), output_format);
			return EXIT_FAILURE;
		}
		rdf_format = enum_value->value;

		stmt = describe_statement_for_urns (connection, urns);
		main_loop = g_main_loop_new (NULL, FALSE);
		tracker_sparql_statement_serialize_async (stmt,
		                                          TRACKER_SERIALIZE_FLAGS_NONE,
		                                          rdf_format,
		                                          NULL,
		                                          serialize_cb,
		                                          main_loop);
		g_main_loop_run (main_loop);
	} else {
		g_autoptr (TrackerSparqlStatement) stmt = NULL;
		g_autoptr (TrackerSparqlCursor) cursor = NULL;
		g_autoptr (GMainLoop) main_loop = NULL;
		TrackerNamespaceManager *namespaces;

		namespaces = tracker_sparql_connection_get_namespace_manager (connection);
		stmt = describe_statement_for_urns (connection, urns);
		cursor = tracker_sparql_statement_execute (stmt, NULL, NULL);

		print_plain (cursor, namespaces);
	}

 out:
	g_list_free_full (urns, g_free);

	tracker_term_pager_close ();

	return EXIT_SUCCESS;
}

static int
info_run_eligible (void)
{
	char **p;
	g_autoptr (GError) error = NULL;

	for (p = filenames; *p; p++) {
		output_eligible_status_for_file (*p, &error);

		if (error) {
			g_printerr ("%s: %s\n",
			            _("Could not get eligible status: "),
			            error->message);
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}


static gboolean
info_options_enabled (void)
{
	return INFO_OPTIONS_ENABLED ();
}

int
tracker_info (int          argc,
              const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree char *help = NULL;

	output_is_tty = tracker_term_is_tty ();

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	context = g_option_context_new (NULL);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_set_summary (context, _("Retrieve information available for files and resources"));

	inside_build_tree = tracker_cli_check_inside_build_tree (argv[0]);

	argv[0] = "localsearch info";

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		return EXIT_FAILURE;
	}

	if (info_options_enabled ()) {
		if (eligible)
			return info_run_eligible ();

		return info_run ();
	}

	help = g_option_context_get_help (context, TRUE, NULL);
	g_printerr ("%s\n", help);

	return EXIT_FAILURE;
}
