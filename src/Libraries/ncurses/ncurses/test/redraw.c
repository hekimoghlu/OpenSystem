/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
 * $Id: redraw.c,v 1.8 2012/12/08 20:46:02 tom Exp $
 *
 * Demonstrate the redrawwin() and wredrawln() functions.
 * Thomas Dickey - 2006/11/4
 */

#include <test.priv.h>

static void
trash(int beg_x, int max_x, int cur_x)
{
    int x;

    for (x = cur_x; x > beg_x; --x) {
	putchar('\b');
    }
    for (x = beg_x; x < max_x; ++x) {
	if (x < cur_x)
	    putchar('<');
	else if (x == cur_x)
	    putchar('=');
	else if (x > cur_x)
	    putchar('>');
    }
    for (x = max_x; x > cur_x; --x) {
	putchar('\b');
    }
}

static void
test_redraw(WINDOW *win)
{
    WINDOW *win1;
    WINDOW *win2;
    bool done = FALSE;
    int ch, y, x;
    int max_y, max_x;
    int beg_y, beg_x;

    assert(win != 0);

    scrollok(win, TRUE);
    keypad(win, TRUE);
    getmaxyx(win, max_y, max_x);
    getbegyx(win, beg_y, beg_x);
    while (!done) {
	ch = wgetch(win);
	getyx(win, y, x);
	switch (ch) {
	case 'q':
	    /* FALLTHRU */
	case ESCAPE:
	    done = TRUE;
	    break;
	case 'w':
	    win1 = newwin(max_y, max_x,
			  beg_y, beg_x);
	    win2 = newwin(max_y - 2, max_x - 2,
			  beg_y + 1, beg_x + 1);
	    box(win1, 0, 0);
	    wrefresh(win1);

	    test_redraw(win2);

	    delwin(win2);
	    delwin(win1);

	    touchwin(win);
	    break;

	case '!':
	    /*
	     * redrawwin() and wredrawln() do not take into account the
	     * possibility that the cursor may have moved.  That makes them
	     * cumbersome for using with a shell command.  So we simply
	     * trash the current line of the window using backspace/overwrite.
	     */
	    trash(beg_x, max_x, x + beg_x);
	    break;

#ifdef NCURSES_VERSION
	case '@':
	    /*
	     * For a shell command, we can work around the problem noted above
	     * using mvcur().  It is ifdef'd for NCURSES, since X/Open does
	     * not define the case where the old location is unknown. 
	     */
	    IGNORE_RC(system("date"));
	    mvcur(-1, -1, y, x);
	    break;
#endif

	case CTRL('W'):
	    redrawwin(win);
	    break;

	case CTRL('L'):
	    wredrawln(win, y, 1);
	    break;

	case KEY_UP:
	    if (y > 0)
		wmove(win, y - 1, x);
	    break;

	case KEY_DOWN:
	    if (y < max_y)
		wmove(win, y + 1, x);
	    break;

	case KEY_LEFT:
	    if (x > 0)
		wmove(win, y, x - 1);
	    break;

	case KEY_RIGHT:
	    if (x < max_x)
		wmove(win, y, x + 1);
	    break;

	default:
	    if (ch > KEY_MIN) {
		waddstr(win, keyname(ch));
	    } else {
		waddstr(win, unctrl(UChar(ch)));
	    }
	    break;
	}
	wnoutrefresh(win);
	doupdate();
    }
}

int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    initscr();
    raw();
    noecho();
    test_redraw(stdscr);
    endwin();
    ExitProgram(EXIT_SUCCESS);
}
