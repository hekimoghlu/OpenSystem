/*
 * Copyright (C) 2025, Red Hat Inc.
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
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include "tracker-inhibit.h"

#include <glib.h>
#include <glib/gi18n.h>
#include <locale.h>

#include "tracker-color.h"
#include "tracker-indexer-proxy.h"
#include "tracker-term-utils.h"

static gchar **command;
static gboolean list;

static GOptionEntry entries[] = {
	{ "list", 'l', 0, G_OPTION_ARG_NONE, &list,
	  N_("List inhibitions"),
	  NULL
	},
	/* Main arguments, the search terms */
	{ G_OPTION_REMAINING, 0, 0,
	  G_OPTION_ARG_STRING_ARRAY, &command,
	  N_("command"),
	  N_("COMMAND")
	},
	{ NULL }
};

static void
print_pause_details (GStrv pause_apps,
                     GStrv pause_reasons)
{
	gint i, cols, col_len[2];
	g_autofree char *col_header1 = NULL, *col_header2 = NULL;

	if (!pause_apps[0] || !pause_reasons[0])
		return;

	tracker_term_dimensions (&cols, NULL);
	col_len[0] = cols / 2;
	col_len[1] = cols / 2 - 1;

	col_header1 = tracker_term_ellipsize (_("Application"), col_len[0], TRACKER_ELLIPSIZE_END);
	col_header2 = tracker_term_ellipsize (_("Reason"), col_len[1], TRACKER_ELLIPSIZE_END);
	g_print (BOLD_BEGIN "%-*s %-*s" BOLD_END "\n",
	         col_len[0], col_header1,
	         col_len[1], col_header2);

	for (i = 0; pause_apps[i] && pause_reasons[i]; i++) {
		g_print ("%-*s %-*s\n",
		         col_len[0], pause_apps[i],
		         col_len[1], pause_reasons[i]);
	}

	g_print ("\n");
}

int
tracker_inhibit (int          argc,
                 const char **argv)
{
	g_autoptr (GOptionContext) context = NULL;
	g_autoptr (TrackerIndexerMiner) indexer_proxy = NULL;
	g_autoptr (GError) error = NULL;
	g_autofree char *help = NULL;
	int cookie = 0;

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	context = g_option_context_new (NULL);
	g_option_context_add_main_entries (context, entries, NULL);
	g_option_context_set_summary (context, _("Inhibit indexing temporarily"));

	argv[0] = "localsearch inhibit";

	indexer_proxy =
		tracker_indexer_miner_proxy_new_for_bus_sync (G_BUS_TYPE_SESSION,
		                                              G_DBUS_PROXY_FLAGS_DO_NOT_AUTO_START |
		                                              G_DBUS_PROXY_FLAGS_DO_NOT_CONNECT_SIGNALS,
		                                              "org.freedesktop.LocalSearch3",
		                                              "/org/freedesktop/Tracker3/Miner/Files",
		                                              NULL, &error);
	if (!indexer_proxy)
		goto error;

	if (!g_option_context_parse (context, &argc, (char***) &argv, &error)) {
		g_printerr ("%s, %s\n", _("Unrecognized options"), error->message);
		help = g_option_context_get_help (context, FALSE, NULL);
		g_printerr ("%s\n", help);

		return EXIT_FAILURE;
	} else if (list) {
		g_auto (GStrv) apps = NULL, reasons = NULL;

		if (tracker_indexer_miner_call_get_pause_details_sync (indexer_proxy,
		                                                       &apps,
		                                                       &reasons,
		                                                       NULL, NULL))
			print_pause_details (apps, reasons);

		return EXIT_SUCCESS;
	} else if (!command) {
		help = g_option_context_get_help (context, FALSE, NULL);
		g_printerr ("%s\n", help);
		return EXIT_FAILURE;
	}

	if (!tracker_indexer_miner_call_pause_for_process_sync (indexer_proxy,
	                                                        command[0],
	                                                        _("Indexing inhibited through command line"),
	                                                        &cookie,
	                                                        NULL,
	                                                        &error))
		goto error;

	if (!g_spawn_sync (NULL, command, NULL,
	                   G_SPAWN_SEARCH_PATH |
	                   G_SPAWN_CHILD_INHERITS_STDIN |
	                   G_SPAWN_CHILD_INHERITS_STDOUT |
	                   G_SPAWN_CHILD_INHERITS_STDERR,
	                   NULL, NULL, NULL, NULL, NULL,
	                   &error))
		goto error;

	if (!tracker_indexer_miner_call_resume_sync (indexer_proxy,
	                                             cookie,
	                                             NULL,
	                                             &error))
		goto error;

	return EXIT_SUCCESS;

 error:
	g_printerr ("%s: %s\n", _("Failed to inhibit indexer"), error->message);

	return EXIT_FAILURE;
}
