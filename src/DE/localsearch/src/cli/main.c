/*
 * Copyright (C) 2014, Lanedo <martyn@lanedo.com>
 * Copyright (C) 2024, Sam Thursfield <sam@afuera.me.uk>
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
#include <locale.h>

#include <glib.h>
#include <glib/gi18n.h>
#include <gio/gio.h>
#include <gio/gdesktopappinfo.h>

#include <tracker-common.h>

#include "tracker-extract.h"
#include "tracker-help.h"
#include "tracker-index.h"
#include "tracker-info.h"
#include "tracker-inhibit.h"
#include "tracker-reset.h"
#include "tracker-search.h"
#include "tracker-status.h"
#include "tracker-tag.h"

const char usage_string[] =
	"localsearch [--version] [--help]\n"
	"            <command> [<args>]";

const char about[] =
	"LocalSearch " PACKAGE_VERSION "\n"
	"\n"
	"This program is free software and comes without any warranty.\n"
	"It is licensed under version 2 or later of the General Public "
	"License which can be viewed at:\n"
	"\n"
	"  http://www.gnu.org/licenses/gpl.txt"
	"\n";

struct cmd_struct {
	const char *cmd;
	int (*fn)(int, const char **);
	const char *help;
};

static int launch_external_command (int argc, const char **argv);

static void print_usage (void);

static struct cmd_struct commands[] = {
	{ "help", tracker_help, N_("Show help on subcommands") },
	{ "extract", tracker_extract, N_("Show metadata extractor output") },
	{ "index", tracker_index, N_("List and change indexed folders") },
	{ "info", tracker_info, N_("Retrieve information available for files and resources") },
	{ "inhibit", tracker_inhibit, N_("Inhibit indexing temporarily") },
	{ "reset", tracker_reset, N_("Erase the indexed data") },
	{ "search", tracker_search, N_("Search for content") },
	{ "status", tracker_status, N_("Provide status and statistics on the data indexed") },
	{ "tag", tracker_tag, N_("Add, remove and list tags") },
	{ "test-sandbox", launch_external_command, N_("Sandbox for a testing environment") },
};

static int
launch_external_command (int          argc,
                         const char **argv)
{
	const char *execdir, *subcommand;
	char *path, *basename;

	execdir = g_getenv ("LOCALSEARCH_CLI_PATH");
	if (!execdir)
		execdir = PYTHON_UTILS_DIR;

	/* Execute subcommand binary */
	subcommand = argv[0];
	basename = g_strdup_printf("localsearch3-%s", subcommand);
	path = g_build_filename (execdir, basename, NULL);

	/* Manipulate argv in place, in order to launch subcommand */
	argv[0] = path;
	return execv (path, (char * const *) argv);
}

static int
print_version (void)
{
	puts (about);
	return 0;
}

static inline void
mput_char (char c, unsigned int num)
{
      while (num--) {
              putchar (c);
      }
}

static void
print_usage_list_cmds (void)
{
        guint longest = 0;
        guint i;

        puts (_("Available localsearch commands are:"));

        for (i = 0; i < G_N_ELEMENTS (commands); i++) {
                if (longest < strlen (commands[i].cmd))
                        longest = strlen (commands[i].cmd);
        }

        for (i = 0; i < G_N_ELEMENTS (commands); i++) {
                g_print ("   %s   ", commands[i].cmd);
                mput_char (' ', longest - strlen (commands[i].cmd));
                puts (_(commands[i].help));
        }
}

static void
print_usage (void)
{
	g_print ("usage: %s\n\n", usage_string);
	print_usage_list_cmds ();
	g_print ("\n%s\n", _("See “localsearch help <command>” to read about a specific subcommand."));
}

int
main (int argc, char *argv[])
{
        int (* func) (int, const char *[]) = NULL;
        const gchar *subcommand = argv[1];
	int i;

	setlocale (LC_ALL, "");

	bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
	textdomain (GETTEXT_PACKAGE);

	if (argc == 1) {
		/* The user didn't specify a command; give them help */
		print_usage ();
		exit (EXIT_SUCCESS);
	}

	if (g_strcmp0 (subcommand, "--version") == 0) {
		print_version ();
		exit (EXIT_SUCCESS);
	} else if (g_strcmp0 (subcommand, "--help") == 0) {
		subcommand = "help";
	}

	if (g_strcmp0 (subcommand, "help") == 0 && argc == 2) {
		/* Print usage here to avoid duplicating it in tracker-help.c */
		print_usage ();
		exit (EXIT_SUCCESS);
	}

	for (i = 0; i < G_N_ELEMENTS (commands); i++) {
		if (g_strcmp0 (commands[i].cmd, subcommand) == 0)
			func = commands[i].fn;
	}

        if (func) {
                return func (argc - 1, (const char **) &argv[1]);
        } else {
                g_printerr (_("“%s” is not a localsearch command. See “localsearch --help”"), subcommand);
                g_printerr ("\n");
                return EXIT_FAILURE;
        }

	return EXIT_FAILURE;
}
