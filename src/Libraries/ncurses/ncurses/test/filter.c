/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
/*
 * Author:  Thomas E. Dickey <dickey@clark.net> 1998
 *
 * $Id: filter.c,v 1.16 2014/08/09 22:35:51 tom Exp $
 */
#include <test.priv.h>

#if HAVE_FILTER

/*
 * An example of the 'filter()' function in ncurses, this program prompts
 * for commands and executes them (like a command shell).  It illustrates
 * how ncurses can be used to implement programs that are not full-screen.
 *
 * Ncurses differs slightly from SVr4 curses.  The latter does not flush its
 * state when exiting program mode, so the attributes on the command lines of
 * this program 'bleed' onto the executed commands.  Rather than use the
 * reset_shell_mode() and reset_prog_mode() functions, we could invoke endwin()
 * and refresh(), but that does not work any better.
 */

static int
new_command(char *buffer, int length, int underline)
{
    int code;

    attron(A_BOLD);
    printw("Command: ");
    attron(underline);
    code = getnstr(buffer, length);
    /*
     * If this returns anything except ERR/OK, it would be one of ncurses's
     * extensions.  Fill the buffer with something harmless that the shell
     * will execute as a comment.
     */
#ifdef KEY_EVENT
    if (code == KEY_EVENT)
	strcpy(buffer, "# event!");
#endif
#ifdef KEY_RESIZE
    if (code == KEY_RESIZE) {
	strcpy(buffer, "# resize!");
	getch();
    }
#endif
    attroff(underline);
    attroff(A_BOLD);
    printw("\n");

    return code;
}

static void
usage(void)
{
    static const char *msg[] =
    {
	"Usage: filter [options]"
	,""
	,"Options:"
	,"  -i   use initscr() rather than newterm()"
    };
    unsigned n;
    for (n = 0; n < SIZEOF(msg); n++)
	fprintf(stderr, "%s\n", msg[n]);
    ExitProgram(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    int ch;
    char buffer[80];
    int underline;
    bool i_option = FALSE;

    setlocale(LC_ALL, "");

    while ((ch = getopt(argc, argv, "i")) != -1) {
	switch (ch) {
	case 'i':
	    i_option = TRUE;
	    break;
	default:
	    usage();
	}
    }

    printf("starting filter program using %s...\n",
	   i_option ? "initscr" : "newterm");
    filter();
    if (i_option) {
	initscr();
    } else {
	(void) newterm((char *) 0, stdout, stdin);
    }
    cbreak();
    keypad(stdscr, TRUE);

    if (has_colors()) {
	int background = COLOR_BLACK;
	start_color();
#if HAVE_USE_DEFAULT_COLORS
	if (use_default_colors() != ERR)
	    background = -1;
#endif
	init_pair(1, COLOR_CYAN, (short) background);
	underline = COLOR_PAIR(1);
    } else {
	underline = A_UNDERLINE;
    }

    while (new_command(buffer, sizeof(buffer) - 1, underline) != ERR
	   && strlen(buffer) != 0) {
	reset_shell_mode();
	printf("\n");
	fflush(stdout);
	IGNORE_RC(system(buffer));
	reset_prog_mode();
	touchwin(stdscr);
	erase();
	refresh();
    }
    printw("done");
    refresh();
    endwin();
    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(void)
{
    printf("This program requires the filter function\n");
    ExitProgram(EXIT_FAILURE);
}
#endif /* HAVE_FILTER */
