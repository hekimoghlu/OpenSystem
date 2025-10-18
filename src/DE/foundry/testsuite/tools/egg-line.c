/* egg-line.c
 *
 * Copyright 2009 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <stdio.h>

#include <readline/readline.h>
#include <readline/history.h>

#include "egg-line.h"

struct _EggLine
{
	EggLineCommand *commands;
  EggLineMissing  missing;
	char           *prompt;
  guint           quit : 1;
};

static EggLineCommand empty[] = {
	{ NULL }
};

static EggLine *current = NULL;

static void
egg_line_finalize (gpointer data)
{
  EggLine *self = data;

  self->commands = NULL;
  self->missing = NULL;

  g_clear_pointer (&self->prompt, g_free);
}

EggLine *
egg_line_new (void)
{
  EggLine *self = g_atomic_rc_box_new0 (EggLine);

	self->prompt = g_strdup ("> ");

  return self;
}

static gchar*
egg_line_generator (const gchar *text,
                    gint         state)
{
	EggLineCommand  *command;
	static gint      list_index,
	                 len   = 0,
	                 argc  = 0;
	const gchar     *name;
	gchar           *tmp,
	               **argv  = NULL,
	               **largv = NULL;

	if (!current || !text || !current->commands)
		return NULL;

	command = egg_line_resolve (current, rl_line_buffer, &argc, &argv);
	largv = argv;

	if (command) {
		if (command->generator)
			command = command->generator (current, &argc, &argv);
		else
			command = empty;
	}
	else {
		command = current->commands;
	}

	if (argv && argv[0])
		tmp = g_strdup (argv[0]);
	else
		tmp = g_strdup ("");

	g_strfreev (largv);

	if (!state)
		list_index = 0;

	len = strlen (tmp);

	while (NULL != (name = command[list_index].name)) {
		list_index++;
		if ((g_ascii_strncasecmp (name, tmp, len) == 0)) {
			return g_strdup (name);
		}
	}

	return NULL;
}

static gchar**
egg_line_completion (const gchar *text,
                     gint         start,
                     gint         end)
{
	return rl_completion_matches (text, egg_line_generator);
}

/**
 * egg_line_quit:
 * @self: An #EggLine
 *
 * Quits the readline loop after the current line has completed.
 */
void
egg_line_quit (EggLine *self)
{
	g_return_if_fail (self != NULL);

	self->quit = TRUE;
}

/**
 * egg_line_run:
 * @self: A #EggLine
 *
 * Blocks running the readline interaction using stdin and stdout.
 */
void
egg_line_run (EggLine *self)
{
	char *text;

	g_return_if_fail (self != NULL);

	current = self;

	self->quit = FALSE;

	rl_readline_name = "egg-line";
	rl_attempted_completion_function = egg_line_completion;

	while (!self->quit) {
		text = readline (self->prompt);

		if (!text)
			break;

		if (*text) {
			add_history (text);
			egg_line_execute (self, text);
		}
	}

	g_print ("\n");
	current = NULL;
}

/**
 * egg_line_set_commands:
 * @self: A #EggLine
 * @entries: A %NULL terminated array of #EggLineCommand
 *
 * Sets the top-level set of #EggLineCommand<!-- -->'s to be completed
 * during runtime.
 */
void
egg_line_set_commands (EggLine              *self,
                       const EggLineCommand *entries)
{
	g_return_if_fail (self != NULL);

	self->commands = (EggLineCommand*) entries;
}

/**
 * egg_line_set_prompt:
 * @self: An #EggLine
 * @prompt: a string containing the prompt
 *
 * Sets the line prompt.
 */
void
egg_line_set_prompt (EggLine    *self,
                     const char *prompt)
{
	g_return_if_fail (self != NULL);
	g_return_if_fail (prompt != NULL);

  g_set_str (&self->prompt, prompt);
}

/**
 * egg_line_execute:
 * @self: An #EggLine
 * @text: the command to execute
 *
 * Executes the command as described by @text.
 */
void
egg_line_execute (EggLine     *self,
                  const gchar *text)
{
	EggLineStatus     result;
	EggLineCommand   *command;
	GError           *error = NULL;
	gchar           **argv  = NULL;
	gint              argc  = 0;

	g_return_if_fail (self != NULL);
	g_return_if_fail (text != NULL);

	command = egg_line_resolve (self, text, &argc, &argv);

	if (command && command->callback) {
		result = command->callback (self, command, argc, argv, &error);
		switch (result) {
		case EGG_LINE_STATUS_OK:
			break;
		case EGG_LINE_STATUS_BAD_ARGS:
			egg_line_show_usage (self, command);
			break;
		case EGG_LINE_STATUS_FAILURE:
			g_printerr ("EGG_LINE_ERROR: %s\n", error->message);
			g_error_free (error);
			break;
		default:
			break;
		}
	}
	else if (command && command->usage) {
		egg_line_show_usage (self, command);
	}
	else {
    if (self->missing)
      self->missing (self, text);
	}

	g_strfreev (argv);
}

/**
 * egg_line_resolve:
 * @self: An #EggLine
 * @text: command text
 *
 * Resolves a command and arguments for @text.
 *
 * Return value: the instance of #EggLineCommand.  This value should not be
 *   modified or freed.
 */
EggLineCommand*
egg_line_resolve (EggLine       *self,
                  const gchar   *text,
                  gint          *argc,
                  gchar       ***argv)
{
	EggLineCommand  *command = NULL,
	                *tmp     = NULL,
	                *result  = NULL;
	gchar          **largv   = NULL,
	               **origv   = NULL;
	gint             largc   = 0,
	                 i;
	GError          *error   = NULL;

	g_return_val_if_fail (self != NULL, NULL);
	g_return_val_if_fail (text != NULL, NULL);

	if (argc)
		*argc = 0;

	if (argv)
		*argv = NULL;

	if (strlen (text) == 0)
		return NULL;

	if (!g_shell_parse_argv (text, &largc, &largv, &error)) {
		g_printerr ("%s\n", error->message);
		g_error_free (error);
		return NULL;
	}

	if (self->commands == NULL)
		return NULL;

	command = self->commands;
	origv = largv;

	for (i = 0; largv[0] && command[i].name;) {
		if (g_str_equal (largv[0], command[i].name)) {
			if (command[i].generator) {
				tmp = command[i].generator (self, &largc, &largv);
			}

			result = &command[i];
			command = tmp ? tmp : empty;

			i = 0;
			largv = &largv[1];
			largc--;
		}
		else i++;
	}

	if (argv)
		*argv = largv ? g_strdupv (largv) : NULL;

	if (argc)
		*argc = largc;

	g_strfreev (origv);

	return result;
}

/**
 * egg_line_show_usage:
 * @self: An #EggLine
 * @command: An #EggLineCommand
 *
 * Shows command usage for @command.
 */
void
egg_line_show_usage (EggLine              *self,
                     const EggLineCommand *command)
{
	g_return_if_fail (self != NULL);
	g_return_if_fail (command != NULL);

	g_print ("usage: %s\n", command->usage ? command->usage : "");
}

EggLine *
egg_line_ref (EggLine *self)
{
  return g_atomic_rc_box_acquire (self);
}

void
egg_line_unref (EggLine *self)
{
  g_atomic_rc_box_release_full (self, egg_line_finalize);
}

void
egg_line_set_missing_handler (EggLine        *self,
                              EggLineMissing  missing)
{
  g_return_if_fail (self != NULL);

  self->missing = missing;
}
