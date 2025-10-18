/*
 * Copyright (C) 2015-2016, Sam Thursfield <sam@afuera.me.uk>
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

#include <glib.h>
#include <glib/gi18n.h>
#include <gio/gio.h>
#include <locale.h>

#include <tracker-common.h>

#include "tracker-cli-utils.h"

static gboolean inside_build_tree = FALSE;
static gchar *output_format = "turtle";
static gchar **filenames;

#define EXTRACT_OPTIONS_ENABLED()	  \
	((filenames && g_strv_length (filenames) > 0))

static GOptionEntry entries[] = {
	{ "output-format", 'o', 0, G_OPTION_ARG_STRING, &output_format,
	  N_("Output results format: “turtle”, “trig” or “json-ld”"),
	  N_("FORMAT") },
	{ G_OPTION_REMAINING, 0, 0, G_OPTION_ARG_FILENAME_ARRAY, &filenames,
	  N_("FILE"),
	  N_("FILE") },
	{ NULL }
};

static void
extractor_child_setup (gpointer user_data)
{
#ifdef HAVE_LANDLOCK
	g_autofree gchar *folder = NULL;

	folder = g_path_get_dirname (user_data);

	if (!tracker_landlock_init ((const gchar*[]) { folder, NULL }))
		g_assert_not_reached ();
#endif
}

static gint
extract_files (char *output_format)
{
	g_autofree char *tracker_extract_path = NULL;
	g_autoptr (GError) error = NULL;
	char **p;

	tracker_term_pipe_to_pager ();

	if (inside_build_tree) {
		/* Developer convenience - use uninstalled version if running from build tree */
		tracker_extract_path = g_build_filename(BUILDROOT, "src", "extractor", "localsearch-extractor-3", NULL);
	} else {
		tracker_extract_path = g_build_filename(LIBEXECDIR, "localsearch-extractor-3", NULL);
	}

	for (p = filenames; *p; p++) {
		char *argv[] = {tracker_extract_path,
		                "--output-format", output_format,
		                "--file", *p, NULL };

		g_spawn_sync (NULL, argv, NULL, G_SPAWN_DEFAULT,
		              extractor_child_setup, *p,
		              NULL, NULL, NULL, &error);

		if (error) {
			g_printerr ("%s: %s\n",
			            _("Could not run tracker-extract: "),
			            error->message);
			return EXIT_FAILURE;
		}
	}

	tracker_term_pager_close ();

	return EXIT_SUCCESS;
}

static int
extract_run (void)
{
	return extract_files (output_format);
}

static gboolean
extract_options_enabled (void)
{
	return EXTRACT_OPTIONS_ENABLED ();
}

int
tracker_extract (int          argc,
                 const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree char *help = NULL;

	context = g_option_context_new (NULL);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_set_summary (context, _("Extract metadata from a file"));

	inside_build_tree = tracker_cli_check_inside_build_tree (argv[0]);

	argv[0] = "localsearch extract";

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		return EXIT_FAILURE;
	}

	if (extract_options_enabled ()) {
		return extract_run ();
	}

	help = g_option_context_get_help (context, TRUE, NULL);
	g_printerr ("%s\n", help);

	return EXIT_FAILURE;
}
