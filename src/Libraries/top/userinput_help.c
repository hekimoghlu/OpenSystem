/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "userinput_help.h"
#include "preferences.h"
#include "top.h"
#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
	int yoffset;
	WINDOW *window;
	PANEL *panel;
	WINDOW *state;
	WINDOW *command;
	WINDOW *description;
} help = {
	.yoffset = 0, .window = NULL, .panel = NULL, .state = NULL, .command = NULL, .description = NULL
};

static void
help_completion(void *tinst, struct user_input_state *s)
{
	user_input_set_state(NULL);
}

static void
help_draw_text_line(const char *statestr, const char *cmdstr, const char *dstr)
{
	mvwaddstr(help.state, help.yoffset, 0, statestr);
	mvwaddstr(help.command, help.yoffset, 0, cmdstr);
	mvwaddstr(help.description, help.yoffset, 0, dstr);

	help.yoffset += 1;
}

static void
help_draw_text(void)
{
	char buf[32];
	const char *user;

	wattron(help.state, A_UNDERLINE);
	mvwaddstr(help.state, 0, 0, "State");
	wattroff(help.state, A_UNDERLINE);

	wattron(help.command, A_UNDERLINE);
	mvwaddstr(help.command, 0, 0, "Command");
	wattroff(help.command, A_UNDERLINE);

	wattron(help.description, A_UNDERLINE);
	mvwaddstr(help.description, 0, 0, "Description");
	wattroff(help.description, A_UNDERLINE);

	help.yoffset += 1;

	help_draw_text_line("", "?", "Display this help screen.");
	help_draw_text_line(
			top_prefs_get_mode_string(), "c<mode>", "Set event counting mode to {a|d|e|n}.");
	help_draw_text_line(
			top_prefs_get_sort_string(), "o<key>", "Set primary sort key to <key>: [+-]keyname.");
	help_draw_text_line("", "", "Keyname may be:{pid|command|cpu|csw|time|threads|");
	help_draw_text_line("", "", "ports|mregion|rprvt|rshrd|rsize|psize|vsize|vprvt|");
	help_draw_text_line("", "", "pgrp|ppid|state|uid|wq|faults|cow|user|msgsent|");
	help_draw_text_line("", "", "msgrecv|sysbsd|sysmach|pageins}.");
	help_draw_text_line(top_prefs_get_secondary_sort_string(), "O<skey>",
			"Set secondary sort key to <skey> (see o<key>).");
	help_draw_text_line("", "q", "Quit top.");
	help_draw_text_line("", "S<sig>\\n<pid>", "Send signal <sig> to pid <pid>.");

	if (-1 == snprintf(buf, sizeof(buf), "%d", top_prefs_get_sleep()))
		buf[0] = '\0';

	help_draw_text_line(buf, "s<delay>", "Set the delay between updates to <delay> seconds.");

	help_draw_text_line(
			top_prefs_get_mmr() ? "on" : "off", "r", "Toggle the memory map reporting.");

	user = top_prefs_get_user();

	help_draw_text_line(
			user ? user : "", "U<user>", "Only display processes owned by <user>, or all.");

	mvwaddstr(help.window, LINES - 1, 0, "Press any key to continue...");
}

static void
help_cleanup(void)
{
	if (help.panel) {
		del_panel(help.panel);
		help.panel = NULL;
	}

	if (help.window) {
		delwin(help.window);
		help.window = NULL;
	}

	if (help.state) {
		delwin(help.state);
		help.state = NULL;
	}

	if (help.command) {
		delwin(help.command);
		help.command = NULL;
	}

	if (help.description) {
		delwin(help.description);
		help.description = NULL;
	}
}

static void
help_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	int c;

	do {
		help.yoffset = 0;
		help.window = newwin(LINES, COLS, 0, 0);

		if (NULL == help.window) {
			user_input_set_state(NULL);
			return;
		}

		help.panel = new_panel(help.window);

		if (NULL == help.panel) {
			help_cleanup();
			user_input_set_state(NULL);
			return;
		}

		help.state = derwin(help.window, LINES - 1, 10, /*y*/ 0, /*x*/ 0);
		help.command = derwin(help.window, LINES - 1, 15, /*y*/ 0,
				/*x*/ 10);
		help.description = derwin(help.window, LINES - 1, COLS - 25, /*y*/ 0,
				/*x*/ 25);

		if (NULL == help.state || NULL == help.command || NULL == help.description) {
			help_cleanup();
			user_input_set_state(NULL);
			return;
		}

		help_draw_text();

		/* Wait for a key press. */
		wtimeout(help.window, -1);
		c = wgetch(help.window);

		help_cleanup();
	} while (KEY_RESIZE == c);

	user_input_set_state(NULL);
}

struct user_input_state top_user_input_help_state
		= { .offset = 0, .completion = help_completion, .draw = help_draw };
