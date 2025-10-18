/*
 * Copyright (C) 2009, Nokia <ivan.frade@nokia.com>
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

#include <tinysparql.h>

#include <tracker-common.h>

#include "tracker-color.h"

static gint limit = -1;
static gint offset;
static gchar **terms;
static gboolean detailed;
static gboolean all;
static gboolean files;
static gboolean folders;
static gboolean music_albums;
static gboolean music_artists;
static gboolean audio_files;
static gboolean image_files;
static gboolean video_files;
static gboolean document_files;
static gboolean software;
static gboolean show_help;

enum {
	ALL,
	DOCUMENTS,
	FILES,
	FOLDERS,
	IMAGES,
	MUSIC_ALBUMS,
	MUSIC_ARTISTS,
	MUSIC,
	SOFTWARE,
	VIDEOS,
	N_QUERIES,
};

static const char *list_queries[] = {
	"/org/freedesktop/LocalSearch/queries/list-all.rq",
	"/org/freedesktop/LocalSearch/queries/list-documents.rq",
	"/org/freedesktop/LocalSearch/queries/list-files.rq",
	"/org/freedesktop/LocalSearch/queries/list-folders.rq",
	"/org/freedesktop/LocalSearch/queries/list-images.rq",
	"/org/freedesktop/LocalSearch/queries/list-music-albums.rq",
	"/org/freedesktop/LocalSearch/queries/list-music-artists.rq",
	"/org/freedesktop/LocalSearch/queries/list-music.rq",
	"/org/freedesktop/LocalSearch/queries/list-software.rq",
	"/org/freedesktop/LocalSearch/queries/list-videos.rq",
};

G_STATIC_ASSERT (G_N_ELEMENTS (list_queries) == N_QUERIES);

static const char *search_queries[] = {
	"/org/freedesktop/LocalSearch/queries/search-all.rq",
	"/org/freedesktop/LocalSearch/queries/search-documents.rq",
	"/org/freedesktop/LocalSearch/queries/search-files.rq",
	"/org/freedesktop/LocalSearch/queries/search-folders.rq",
	"/org/freedesktop/LocalSearch/queries/search-images.rq",
	"/org/freedesktop/LocalSearch/queries/search-music-albums.rq",
	"/org/freedesktop/LocalSearch/queries/search-music-artists.rq",
	"/org/freedesktop/LocalSearch/queries/search-music.rq",
	"/org/freedesktop/LocalSearch/queries/search-software.rq",
	"/org/freedesktop/LocalSearch/queries/search-videos.rq",
};

G_STATIC_ASSERT (G_N_ELEMENTS (search_queries) == N_QUERIES);

static const char *titles[] = {
	N_("Results"),
	N_("Files"),
	N_("Files"),
	N_("Folders"),
	N_("Files"),
	N_("Albums"),
	N_("Artists"),
	N_("Files"),
	N_("Files"),
	N_("Files"),
};

G_STATIC_ASSERT (G_N_ELEMENTS (titles) == N_QUERIES);

static GOptionEntry entries_resource_type[] = {
	/* Search types */
	{ "files", 'f', 0, G_OPTION_ARG_NONE, &files,
	  N_("Search for files"),
	  NULL
	},
	{ "folders", 's', 0, G_OPTION_ARG_NONE, &folders,
	  N_("Search for folders"),
	  NULL
	},
	{ "audio", 'a', 0, G_OPTION_ARG_NONE, &audio_files,
	  N_("Search for audio files"),
	  NULL
	},
	{ "music-albums", 0, 0, G_OPTION_ARG_NONE, &music_albums,
	  N_("Search for music albums"),
	  NULL
	},
	{ "music-artists", 0, 0, G_OPTION_ARG_NONE, &music_artists,
	  N_("Search for music artists"),
	  NULL
	},
	{ "images", 'i', 0, G_OPTION_ARG_NONE, &image_files,
	  N_("Search for image files"),
	  NULL
	},
	{ "videos", 'v', 0, G_OPTION_ARG_NONE, &video_files,
	  N_("Search for video files"),
	  NULL
	},
	{ "documents", 't', 0, G_OPTION_ARG_NONE, &document_files,
	  N_("Search for document files"),
	  NULL
	},
	{ "software", 0, 0, G_OPTION_ARG_NONE, &software,
	  N_("Search for software files"),
	  NULL
	},
	{ NULL }
};

static GOptionEntry entries[] = {
	/* Semantic options */
	{ "limit", 'l', 0, G_OPTION_ARG_INT, &limit,
	  N_("Limit the number of results shown"),
	  NULL
	},
	{ "offset", 'o', 0, G_OPTION_ARG_INT, &offset,
	  N_("Offset the results"),
	  "0"
	},
	{ "detailed", 'd', 0, G_OPTION_ARG_NONE, &detailed,
	  N_("Show URNs for results"),
	  NULL
	},
	{ "all", 'a', 0, G_OPTION_ARG_NONE, &all,
	  N_("Return all non-existing matches too (i.e. include unmounted volumes)"),
	  NULL
	},
	{ "help", 'h', 0, G_OPTION_ARG_NONE, &show_help,
	  N_("Show help options"),
	  NULL,
	},

	/* Main arguments, the search terms */
	{ G_OPTION_REMAINING, 0, 0,
	  G_OPTION_ARG_STRING_ARRAY, &terms,
	  N_("search terms"),
	  N_("TERMS")
	},
	{ NULL }
};

static gchar *
get_fts_string (GStrv search_words)
{
	GString *fts;
	gint i, len;

	if (!search_words) {
		return NULL;
	}

	fts = g_string_new ("");
	len = g_strv_length (search_words);

	for (i = 0; i < len; i++) {
		gchar *escaped;

		/* Properly escape the input string as it's going to be passed
		 * in a sparql query */
		escaped = tracker_sparql_escape_string (search_words[i]);

		g_string_append (fts, escaped);

		if (i < len - 1) {
			g_string_append (fts, " ");
		}

		g_free (escaped);
	}

	return g_string_free (fts, FALSE);
}

static inline void
print_snippet (const gchar *snippet)
{
	if (!snippet || *snippet == '\0') {
		return;
	} else {
		gchar *compressed;
		gchar *p;

		compressed = g_strdup (snippet);

		for (p = compressed;
		     p && *p != '\0';
		     p = g_utf8_next_char (p)) {
			if (*p == '\r' || *p == '\n') {
				/* inline \n and \r */
				*p = ' ';
			}
		}

		g_print ("  %s\n", compressed);
		g_free (compressed);
	}

	g_print ("\n");
}

static gboolean
get_cursor_results (TrackerSparqlCursor *cursor,
                    const char          *name,
                    gint                 search_limit,
                    gboolean             details)
{
	gboolean is_tty;
	gint count = 0;

	is_tty = tracker_term_is_tty ();

	tracker_term_pipe_to_pager ();

	if (is_tty)
		g_print (BOLD_BEGIN "%s:" BOLD_END "\n", name);

	while (tracker_sparql_cursor_next (cursor, NULL, NULL)) {
		if (details) {
			g_print ("%s (%s)\n",
			         tracker_sparql_cursor_get_string (cursor, 1, NULL),
			         tracker_sparql_cursor_get_string (cursor, 0, NULL));

			if (tracker_sparql_cursor_get_n_columns (cursor) > 2)
				print_snippet (tracker_sparql_cursor_get_string (cursor, 2, NULL));
		} else {
			g_print ("%s\n", tracker_sparql_cursor_get_string (cursor, 1, NULL));
		}

		count++;
	}

	g_print ("\n");

	tracker_term_pager_close ();

	return TRUE;
}

static gboolean
query_data (TrackerSparqlConnection *connection,
            const char              *resource_path,
            const char              *name,
            const char              *fts_match,
            gboolean                 show_all,
            gint                     search_offset,
            gint                     search_limit,
            gboolean                 details)
{
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	g_autofree char *fts = NULL;
	g_autoptr (GError) error = NULL;

	stmt = tracker_sparql_connection_load_statement_from_gresource (connection,
	                                                                resource_path,
	                                                                NULL,
	                                                                &error);
	if (error) {
		g_printerr ("%s\n", error->message);
		return FALSE;
	}

	if (fts_match) {
		tracker_sparql_statement_bind_string (stmt, "match", fts_match);
		tracker_sparql_statement_bind_int (stmt, "detailed", detailed);
	}

	tracker_sparql_statement_bind_int (stmt, "showAll", show_all);
	tracker_sparql_statement_bind_int (stmt, "offset", search_offset);
	tracker_sparql_statement_bind_int (stmt, "limit", limit);

	cursor = tracker_sparql_statement_execute (stmt, NULL, &error);

	if (error) {
		g_printerr ("%s\n", error->message);
		return FALSE;
	}

	return get_cursor_results (cursor, name, search_limit, details);
}

static gint
search_run (void)
{
	g_autoptr (TrackerSparqlConnection) connection = NULL;
	g_autofree char *fts = NULL;
	const char *resource_path = NULL, *title = NULL;
	int query_type;
	gboolean success;
	g_autoptr (GError) error = NULL;

	connection = tracker_sparql_connection_bus_new ("org.freedesktop.LocalSearch3",
							NULL, NULL, &error);

	if (!connection) {
		g_printerr ("%s: %s\n",
		            _("Could not connect to LocalSearch"),
		            error->message);
		return EXIT_FAILURE;
	}

	if (files)
		query_type = FILES;
	else if (folders)
		query_type = FOLDERS;
	else if (music_albums)
		query_type = MUSIC_ALBUMS;
	else if (music_artists)
		query_type = MUSIC_ARTISTS;
	else if (audio_files)
		query_type = MUSIC;
	else if (image_files)
		query_type = IMAGES;
	else if (document_files)
		query_type = DOCUMENTS;
	else if (video_files)
		query_type = VIDEOS;
	else
		query_type = ALL;

	fts = get_fts_string (terms);

	resource_path = fts ?
		search_queries[query_type] :
		list_queries[query_type];

	title = titles[query_type];

	success = query_data (connection, resource_path, _(title),
			      fts, all, offset, limit, detailed);

	return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

int
tracker_search (int          argc,
                const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	GOptionGroup *resource_type;
	g_autoptr (GError) error = NULL;
	g_autofree char *help = NULL;

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	argv[0] = "localsearch search";

	context = g_option_context_new (NULL);
        g_option_context_set_summary (context, _("Search for content"));
        resource_type =
	        g_option_group_new ("resource-type",
	                            "Resource Type Options:",
	                            "Show help for resource type options",
	                            NULL, NULL);

        g_option_context_set_help_enabled (context, FALSE);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_add_group (context, resource_type);
	g_option_group_add_entries (resource_type, entries_resource_type);

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		help = g_option_context_get_help (context, FALSE, NULL);
		g_printerr ("%s\n", help);
		return EXIT_FAILURE;
	} else if (show_help) {
		help = g_option_context_get_help (context, FALSE, NULL);
		g_printerr ("%s\n", help);
		return EXIT_SUCCESS;
	}

	return search_run ();
}
