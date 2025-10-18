/*
 * Copyright (C) 2009, Nokia <ivan.frade@nokia.com>
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
#include <time.h>
#include <locale.h>

#include <glib.h>
#include <glib/gi18n.h>
#include <gio/gio.h>

#include <tinysparql.h>

#define TAG_OPTIONS_ENABLED() \
	(resource || \
	 add_tag || \
	 remove_tag || \
	 list)

#define DELETE_TAG_FROM_FILE \
	"/org/freedesktop/LocalSearch/queries/delete-tag-from-file.rq"
#define DELETE_TAG \
	"/org/freedesktop/LocalSearch/queries/delete-tag.rq"
#define GET_FILES_WITH_TAG \
	"/org/freedesktop/LocalSearch/queries/get-files-with-tag.rq"
#define GET_TAGS \
	"/org/freedesktop/LocalSearch/queries/get-tags.rq"
#define GET_TAGS_FOR_FILE \
	"/org/freedesktop/LocalSearch/queries/get-tags-for-file.rq"
#define INSERT_TAG \
	"/org/freedesktop/LocalSearch/queries/insert-tag.rq"
#define INSERT_TAG_WITH_DESC \
	"/org/freedesktop/LocalSearch/queries/insert-tag-with-desc.rq"
#define INSERT_TAG_ON_FILE \
	"/org/freedesktop/LocalSearch/queries/insert-tag-on-file.rq"

static gint limit = 512;
static gint offset;
static gchar **resource;
static gchar *add_tag;
static gchar *remove_tag;
static gchar *description;
static gboolean list;
static gboolean show_resources;

static GOptionEntry entries[] = {
	{ "list", 't', 0, G_OPTION_ARG_NONE, &list,
	  N_("List all tags"),
	  NULL,
	},
	{ "show-files", 's', 0, G_OPTION_ARG_NONE, &show_resources,
	  N_("Show files associated with each tag (this is only used with --list)"),
	  NULL
	},
	{ "add", 'a', 0, G_OPTION_ARG_STRING, &add_tag,
	  N_("Add a tag (if FILEs are omitted, TAG is not associated with any files)"),
	  N_("TAG")
	},
	{ "delete", 'd', 0, G_OPTION_ARG_STRING, &remove_tag,
	  N_("Delete a tag (if FILEs are omitted, TAG is removed for all files)"),
	  N_("TAG")
	},
	{ "description", 'e', 0, G_OPTION_ARG_STRING, &description,
	  N_("Description for a tag (this is only used with --add)"),
	  N_("STRING")
	},
	{ "limit", 'l', 0, G_OPTION_ARG_INT, &limit,
	  N_("Limit the number of results shown"),
	  "512"
	},
	{ "offset", 'o', 0, G_OPTION_ARG_INT, &offset,
	  N_("Offset the results"),
	  "0"
	},
	{ G_OPTION_REMAINING, 0, 0,
	  G_OPTION_ARG_FILENAME_ARRAY, &resource,
	  N_("FILE…")},
	{ NULL }
};

static void
get_all_tags_show_tag_id (TrackerSparqlConnection *connection,
                          const gchar             *id)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                GET_FILES_WITH_TAG,
	                                                                NULL,
	                                                                &error);
	if (stmt) {
		tracker_sparql_statement_bind_string (stmt, "tag", id);
		cursor = tracker_sparql_statement_execute (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not get files related to tag"),
		            error->message);
		g_error_free (error);
		return;
	}

	while (tracker_sparql_cursor_next (cursor, NULL, NULL))
		g_print ("  %s\n", tracker_sparql_cursor_get_string (cursor, 0, NULL));
}

static gboolean
get_all_tags (TrackerSparqlConnection *connection,
              gint                     search_offset,
              gint                     search_limit,
              gboolean                 show_resources)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                GET_TAGS,
	                                                                NULL,
	                                                                &error);
	if (stmt) {
		tracker_sparql_statement_bind_int (stmt, "limit", limit);
		tracker_sparql_statement_bind_int (stmt, "offset", offset);
		cursor = tracker_sparql_statement_execute (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not get all tags"),
		            error->message);
		return FALSE;
	}

	g_print ("%s:\n", _("Tags (shown by name)"));

	while (tracker_sparql_cursor_next (cursor, NULL, NULL)) {
		const gchar *id;
		const gchar *tag;
		const gchar *description;
		gint n_resources = 0;

		id = tracker_sparql_cursor_get_string (cursor, 0, NULL);
		tag = tracker_sparql_cursor_get_string (cursor, 1, NULL);
		description = tracker_sparql_cursor_get_string (cursor, 2, NULL);
		n_resources = tracker_sparql_cursor_get_integer (cursor, 3);

		g_print ("%s", tag);
		if (description)
			g_print (" (%s)", description);
		g_print ("\n");

		if (n_resources > 0) {
			if (show_resources) {
				get_all_tags_show_tag_id (connection, id);
			} else {
				g_print ("  ");
				g_print (g_dngettext (NULL,
				                      "%d file",
				                      "%d files",
				                      n_resources),
				         n_resources);
				g_print ("\n");
			}
		}
	}

	return TRUE;
}

static gboolean
create_tag (TrackerSparqlConnection *connection,
            const gchar             *tag,
            const gchar             *description)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                description ?
	                                                                INSERT_TAG_WITH_DESC :
	                                                                INSERT_TAG,
	                                                                NULL,
	                                                                &error);

	if (stmt) {
		tracker_sparql_statement_bind_string (stmt, "tag", tag);
		if (description)
			tracker_sparql_statement_bind_string (stmt, "desc", description);

		tracker_sparql_statement_update (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not add tag"),
		            error->message);
		return FALSE;
	}

	g_print ("%s\n", _("Tag was added successfully"));

	return TRUE;
}

static gboolean
add_tag_for_urn (TrackerSparqlConnection *connection,
                 const char              *uri,
                 const gchar             *tag)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                INSERT_TAG_ON_FILE,
	                                                                NULL,
	                                                                &error);

	if (stmt) {
		tracker_sparql_statement_bind_string (stmt, "uri", uri);
		tracker_sparql_statement_bind_string (stmt, "tag", tag);
		tracker_sparql_statement_update (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not add tag to files"),
		            error->message);
		return FALSE;
	}

	g_print ("%s\n", _("Tagged"));

	return TRUE;
}

static gboolean
remove_tag_for_urn (TrackerSparqlConnection *connection,
                    const gchar             *uri,
                    const gchar             *tag)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                DELETE_TAG_FROM_FILE,
	                                                                NULL,
	                                                                &error);
	if (stmt) {
		tracker_sparql_statement_bind_string (stmt, "uri", uri);
		tracker_sparql_statement_bind_string (stmt, "tag", tag);
		tracker_sparql_statement_update (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not remove tag"),
		            error->message);
		return FALSE;
	}

	g_print ("%s\n", _("Tag was removed successfully"));

	return TRUE;
}

static gboolean
clear_tag (TrackerSparqlConnection *connection,
           const gchar             *tag)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                DELETE_TAG,
	                                                                NULL,
	                                                                &error);
	if (stmt) {
		tracker_sparql_statement_bind_string (stmt, "tag", tag);
		tracker_sparql_statement_update (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not remove tag"),
		            error->message);
		return FALSE;
	}

	g_print ("%s\n", _("Tag was removed successfully"));

	return TRUE;
}

static gboolean
get_tags_by_file (TrackerSparqlConnection *connection,
                  const gchar             *uri)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                GET_TAGS_FOR_FILE,
	                                                                NULL,
	                                                                &error);
	if (stmt) {
		tracker_sparql_statement_bind_string (stmt, "uri", uri);
		cursor = tracker_sparql_statement_execute (stmt, NULL, &error);
	}

	if (error) {
		g_printerr ("%s, %s\n",
		            _("Could not get all tags"),
		            error->message);
		g_error_free (error);

		return FALSE;
	}

	while (tracker_sparql_cursor_next (cursor, NULL, NULL))
		g_print ("%s\n", tracker_sparql_cursor_get_string (cursor, 1, NULL));

	return TRUE;
}

static int
tag_run (void)
{
	g_autoptr (TrackerSparqlConnection) connection = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree char *uri = NULL;

	connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
	                                                NULL, NULL, &error);

	if (!connection) {
		g_printerr ("%s: %s\n",
		            _("Could not establish a connection to LocalSearch"),
		            error->message);
		return EXIT_FAILURE;
	}

	if (resource && resource[0]) {
		g_autoptr (GFile) file = NULL;

		file = g_file_new_for_commandline_arg (resource[0]);
		uri = g_file_get_uri (file);
	}

	if (list) {
		if (!get_all_tags (connection, offset, limit, show_resources))
			return EXIT_FAILURE;
	} else if (add_tag) {
		if (!create_tag (connection, add_tag, description))
			return EXIT_FAILURE;

		if (uri && !add_tag_for_urn (connection, uri, add_tag))
			return EXIT_FAILURE;
	} else if (remove_tag) {
		if (uri) {
			if (!remove_tag_for_urn (connection, uri, remove_tag))
				return EXIT_FAILURE;
		} else {
			if (!clear_tag (connection, remove_tag))
				return EXIT_FAILURE;
		}
	} else if (uri) {
		if (!get_tags_by_file (connection, uri))
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int
tracker_tag (int          argc,
             const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree char *help = NULL;
	const gchar *failed;

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	context = g_option_context_new (NULL);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_set_summary (context, _("Add, remove and list tags"));

	argv[0] = "localsearch tag";

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		g_error_free (error);
		return EXIT_FAILURE;
	}

	if (!list && show_resources) {
		failed = _("The --list option is required for --show-files");
	} else if (add_tag && remove_tag) {
		failed = _("Add and delete actions can not be used together");
	} else if (description && !add_tag) {
		failed = _("The --description option can only be used with --add");
	} else {
		failed = NULL;
	}

	if (failed) {
		g_printerr ("%s\n\n", failed);
		return EXIT_FAILURE;
	}

	if (!TAG_OPTIONS_ENABLED () || g_strv_length (resource) > 1) {
		help = g_option_context_get_help (context, TRUE, NULL);
		g_printerr ("%s\n", help);
		return EXIT_FAILURE;
	}

	return tag_run ();
}
