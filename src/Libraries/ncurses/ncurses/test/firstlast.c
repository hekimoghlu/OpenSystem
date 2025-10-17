/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
 * This test was written by Alexander V. Lukyanov to demonstrate difference
 * between ncurses 4.1 and SVR4 curses
 *
 * $Id: firstlast.c,v 1.7 2010/05/01 19:11:55 tom Exp $
 */

#include <test.priv.h>

static void
fill(WINDOW *w, const char *str)
{
    const char *s;
    int x0 = -1, y0 = -1;
    int x1, y1;

    for (;;) {
	for (s = str; *s; s++) {
	    getyx(w, y1, x1);
	    if (waddch(w, UChar(*s)) == ERR
		|| (x1 == x0 && y1 == y0)) {
		wmove(w, 0, 0);
		return;
	    }
	    x0 = x1;
	    y0 = y1;
	}
    }
}

int
main(int argc GCC_UNUSED,
     char *argv[]GCC_UNUSED)
{
    WINDOW *large, *small;
    initscr();
    noecho();

    large = newwin(20, 60, 2, 10);
    small = newwin(10, 30, 7, 25);

    /* test 1 - addch */
    fill(large, "LargeWindow");

    refresh();
    wrefresh(large);
    wrefresh(small);

    MvWAddStr(small, 5, 5, "   Test <place to change> String   ");
    wrefresh(small);
    getch();

    touchwin(large);
    wrefresh(large);

    MvWAddStr(small, 5, 5, "   Test <***************> String   ");
    wrefresh(small);

    /* DIFFERENCE! */
    getch();

    /* test 2: erase */
    erase();
    refresh();
    getch();

    /* test 3: clrtoeol */
    werase(small);
    wrefresh(small);
    touchwin(large);
    wrefresh(large);
    wmove(small, 5, 0);
    waddstr(small, " clrtoeol>");
    wclrtoeol(small);
    wrefresh(small);

    /* DIFFERENCE! */ ;
    getch();

    /* test 4: clrtobot */
    werase(small);
    wrefresh(small);
    touchwin(large);
    wrefresh(large);
    wmove(small, 5, 3);
    waddstr(small, " clrtobot>");
    wclrtobot(small);
    wrefresh(small);

    /* DIFFERENCE! */
    getch();

    endwin();

    ExitProgram(EXIT_SUCCESS);
}
