/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include <sys/cdefs.h>

#ifndef __APPLE__
__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)init_disp.c	8.2 (Berkeley) 2/16/94";
#endif
#endif /* __APPLE__ */

/*
 * Initialization code for the display package,
 * as well as the signal handling routines.
 */

#include <sys/stat.h>
#ifdef __APPLE__
#include <sys/ioctl.h>
#endif

#include <err.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>

#include "talk.h"

/*
 * Make sure the callee can write to the screen
 */
void
check_writeable(void)
{
	char *tty;
	struct stat sb;

	if ((tty = ttyname(STDERR_FILENO)) == NULL)
		err(1, "ttyname");
	if (stat(tty, &sb) < 0)
		err(1, "%s", tty);
	if (!(sb.st_mode & S_IWGRP))
		errx(1, "The callee cannot write to this terminal, use \"mesg y\".");
}

/*
 * Set up curses, catch the appropriate signals,
 * and build the various windows.
 */
void
init_display(void)
{
	struct sigaction sa;

	if (initscr() == NULL)
		errx(1, "Terminal type unset or lacking necessary features.");
	(void) sigaction(SIGTSTP, (struct sigaction *)0, &sa);
	sigaddset(&sa.sa_mask, SIGALRM);
	(void) sigaction(SIGTSTP, &sa, (struct sigaction *)0);
	curses_initialized = 1;
	clear();
	refresh();
	noecho();
	crmode();
	signal(SIGINT, sig_sent);
	signal(SIGPIPE, sig_sent);
	signal(SIGWINCH, sig_winch);
	/* curses takes care of ^Z */
	my_win.x_nlines = LINES / 2;
	my_win.x_ncols = COLS;
	my_win.x_win = newwin(my_win.x_nlines, my_win.x_ncols, 0, 0);
	idlok(my_win.x_win, TRUE);
	scrollok(my_win.x_win, TRUE);
	wclear(my_win.x_win);

	his_win.x_nlines = LINES / 2 - 1;
	his_win.x_ncols = COLS;
	his_win.x_win = newwin(his_win.x_nlines, his_win.x_ncols,
	    my_win.x_nlines+1, 0);
	idlok(my_win.x_win, TRUE);
	scrollok(his_win.x_win, TRUE);
	wclear(his_win.x_win);

	line_win = newwin(1, COLS, my_win.x_nlines, 0);
#if defined(hline) || defined(whline) || defined(NCURSES_VERSION)
	whline(line_win, 0, COLS);
#else
	box(line_win, '-', '-');
#endif
	wrefresh(line_win);
	/* let them know we are working on it */
	current_state = "No connection yet";
}

/*
 * Trade edit characters with the other talk. By agreement
 * the first three characters each talk transmits after
 * connection are the three edit characters.
 */
void
set_edit_chars(void)
{
	char buf[3];
	int cc;
	struct termios tio;

	tcgetattr(0, &tio);
	my_win.cerase = tio.c_cc[VERASE];
	my_win.kill = tio.c_cc[VKILL];
	my_win.werase = tio.c_cc[VWERASE];
	if (my_win.cerase == (char)_POSIX_VDISABLE)
		my_win.kill = CERASE;
	if (my_win.kill == (char)_POSIX_VDISABLE)
		my_win.kill = CKILL;
	if (my_win.werase == (char)_POSIX_VDISABLE)
		my_win.werase = CWERASE;
	buf[0] = my_win.cerase;
	buf[1] = my_win.kill;
	buf[2] = my_win.werase;
	cc = write(sockt, buf, sizeof(buf));
	if (cc != sizeof(buf) )
		p_error("Lost the connection");
	cc = read(sockt, buf, sizeof(buf));
	if (cc != sizeof(buf) )
		p_error("Lost the connection");
	his_win.cerase = buf[0];
	his_win.kill = buf[1];
	his_win.werase = buf[2];
}

/* ARGSUSED */
void
sig_sent(int signo __unused)
{

	message("Connection closing. Exiting");
	quit();
}

void
sig_winch(int dummy __unused)
{
 
	gotwinch = 1;
}

/*
 * All done talking...hang up the phone and reset terminal thingy's
 */
void
quit(void)
{

	if (curses_initialized) {
		wmove(his_win.x_win, his_win.x_nlines-1, 0);
		wclrtoeol(his_win.x_win);
		wrefresh(his_win.x_win);
		endwin();
	}
	if (invitation_waiting)
		send_delete();
	exit(0);
}

/*
 * If we get SIGWINCH, recompute both window sizes and refresh things.
 */
void
resize_display(void)
{
	struct winsize ws;

	if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) < 0 ||
	    (ws.ws_row == LINES && ws.ws_col == COLS))
		return;

	/* Update curses' internal state with new window size. */
	resizeterm(ws.ws_row, ws.ws_col);

	/*
	 * Resize each window but wait to refresh the screen until
	 * everything has been drawn so the cursor is in the right spot.
	 */
	my_win.x_nlines = LINES / 2;
	my_win.x_ncols = COLS;
	wresize(my_win.x_win, my_win.x_nlines, my_win.x_ncols);
	mvwin(my_win.x_win, 0, 0);
	clearok(my_win.x_win, TRUE);

	his_win.x_nlines = LINES / 2 - 1;
	his_win.x_ncols = COLS;
	wresize(his_win.x_win, his_win.x_nlines, his_win.x_ncols);
	mvwin(his_win.x_win, my_win.x_nlines + 1, 0);
	clearok(his_win.x_win, TRUE);

	wresize(line_win, 1, COLS);
	mvwin(line_win, my_win.x_nlines, 0);
#if defined(NCURSES_VERSION) || defined(whline)
	whline(line_win, '-', COLS);
#else
	wmove(line_win, my_win.x_nlines, 0);
	box(line_win, '-', '-');
#endif

	/* Now redraw the screen. */
	wrefresh(his_win.x_win);
	wrefresh(line_win);
	wrefresh(my_win.x_win);
}
