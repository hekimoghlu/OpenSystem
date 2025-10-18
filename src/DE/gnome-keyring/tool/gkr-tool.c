/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/* gkr-tool.c: Command line utility

   Copyright (C) 2008 Stefan Walter

   The Gnome Keyring Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The Gnome Keyring Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the Gnome Library; see the file COPYING.LIB.  If not,
   <http://www.gnu.org/licenses/>.

   Author: Stef Walter <stef@memberwebs.com>
*/

#include "config.h"

#include "gkr-tool.h"

#include <glib/gi18n.h>

#include <locale.h>
#include <string.h>

/* -----------------------------------------------------------------------------
 * GENERAL HELPERS
 */

gboolean gkr_tool_mode_quiet = FALSE;

int
gkr_tool_parse_options (int *argc, char** argv[], GOptionEntry *options)
{
	GError *err = NULL;
	GOptionContext *context;
	int ret = 0;
	
	context = g_option_context_new ("- Gnome Keyring Tool");
	g_option_context_add_main_entries (context, options, GETTEXT_PACKAGE);
	
	if (!g_option_context_parse (context, argc, argv, &err)) {
		gkr_tool_handle_error (&err, NULL);
		ret = 2;
	}
	
	g_option_context_free (context);
	return ret;
}

void gkr_tool_handle_error (GError **error, const gchar *format, ...)
{
	gchar *message = NULL;
	GError *err = error ? *error : NULL;
	va_list va;
	
	if (format) {
		va_start(va, format);
		message = g_strdup_vprintf(format, va);
		va_end(va);
	}
	
	g_printerr ("gnome-keyring: %s%s%s\n",
	            message ? message : "",
	            message && err && err->message ? ": " : "",
	            err && err->message ? err->message : "");
	
	g_free (message);
	g_clear_error (error);
}

/* -----------------------------------------------------------------------------
 * COMMAND LINE
 */

typedef struct _CommandInfo {
	const char *name;
	int (*handler) (int argc, char *argv[]);
} CommandInfo;

static CommandInfo command_info[] = {
	{ "certificate-exception", gkr_tool_trust },
	{ "import", gkr_tool_import },
	{ "version", gkr_tool_version },
	{ NULL, NULL }
};

static void
print_general_usage (void)
{
	CommandInfo *cmd;
	const gchar *prefix;
	
	g_printerr (_("usage: gnome-keyring command [options]\n"));
	
	prefix = _("commands: ");
	for (cmd = command_info; cmd->name; ++cmd) {
		g_printerr ("%s%s\n", prefix, cmd->name);
		// Translators: keep same length as translated message "commands: "
		prefix = _("          ");
	}
}

int
main (int argc, char *argv[])
{
	CommandInfo *cmd;
	int ret = -1;

#if !GLIB_CHECK_VERSION(2,35,0)
	g_type_init ();
#endif

	/* internationalisation */
	setlocale (LC_ALL, "");

#ifdef HAVE_GETTEXT
	bindtextdomain (GETTEXT_PACKAGE, GNOMELOCALEDIR);
	textdomain (GETTEXT_PACKAGE);
	bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
#endif

	/* The first argument is the command */
	if (argc < 2) {
		print_general_usage ();
		return 2;
	}
	
	/* Find the command to run */
	for (cmd = command_info; cmd->name; ++cmd) {
		g_return_val_if_fail (argv[1], 1);
		if (strcmp (cmd->name, argv[1]) == 0) {
			/* Remove the command and replace with executable command */
			argv[1] = argv[0];
			ret = (cmd->handler) (argc - 1, argv + 1);
			break;
		}
	}
	
	if (ret == -1) {
		print_general_usage ();
		ret = 2;
	}
	
	return ret;
}
