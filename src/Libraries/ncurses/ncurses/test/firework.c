/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
 * $Id: firework.c,v 1.30 2014/08/02 17:24:07 tom Exp $
 */
#include <test.priv.h>

#include <time.h>

static short my_bg = COLOR_BLACK;

static void
cleanup(void)
{
    curs_set(1);
    endwin();
}

static void
onsig(int n GCC_UNUSED)
{
    cleanup();
    ExitProgram(EXIT_FAILURE);
}

static void
showit(void)
{
    int ch;
    napms(120);
    if ((ch = getch()) != ERR) {
#ifdef KEY_RESIZE
	if (ch == KEY_RESIZE) {
	    erase();
	} else
#endif
	if (ch == 'q') {
	    cleanup();
	    ExitProgram(EXIT_SUCCESS);
	} else if (ch == 's') {
	    nodelay(stdscr, FALSE);
	} else if (ch == ' ') {
	    nodelay(stdscr, TRUE);
	}
    }
}

static short
get_colour(chtype *bold)
{
    int attr;
    attr = (rand() % 16) + 1;

    *bold = A_NORMAL;
    if (attr > 8) {
	*bold = A_BOLD;
	attr &= 7;
    }
    return (short) (attr);
}

static
void
explode(int row, int col)
{
    chtype bold;
    erase();
    MvPrintw(row, col, "-");
    showit();

    init_pair(1, get_colour(&bold), my_bg);
    (void) attrset(AttrArg(COLOR_PAIR(1), bold));
    MvPrintw(row - 1, col - 1, " - ");
    MvPrintw(row + 0, col - 1, "-+-");
    MvPrintw(row + 1, col - 1, " - ");
    showit();

    init_pair(1, get_colour(&bold), my_bg);
    (void) attrset(AttrArg(COLOR_PAIR(1), bold));
    MvPrintw(row - 2, col - 2, " --- ");
    MvPrintw(row - 1, col - 2, "-+++-");
    MvPrintw(row + 0, col - 2, "-+#+-");
    MvPrintw(row + 1, col - 2, "-+++-");
    MvPrintw(row + 2, col - 2, " --- ");
    showit();

    init_pair(1, get_colour(&bold), my_bg);
    (void) attrset(AttrArg(COLOR_PAIR(1), bold));
    MvPrintw(row - 2, col - 2, " +++ ");
    MvPrintw(row - 1, col - 2, "++#++");
    MvPrintw(row + 0, col - 2, "+# #+");
    MvPrintw(row + 1, col - 2, "++#++");
    MvPrintw(row + 2, col - 2, " +++ ");
    showit();

    init_pair(1, get_colour(&bold), my_bg);
    (void) attrset(AttrArg(COLOR_PAIR(1), bold));
    MvPrintw(row - 2, col - 2, "  #  ");
    MvPrintw(row - 1, col - 2, "## ##");
    MvPrintw(row + 0, col - 2, "#   #");
    MvPrintw(row + 1, col - 2, "## ##");
    MvPrintw(row + 2, col - 2, "  #  ");
    showit();

    init_pair(1, get_colour(&bold), my_bg);
    (void) attrset(AttrArg(COLOR_PAIR(1), bold));
    MvPrintw(row - 2, col - 2, " # # ");
    MvPrintw(row - 1, col - 2, "#   #");
    MvPrintw(row + 0, col - 2, "     ");
    MvPrintw(row + 1, col - 2, "#   #");
    MvPrintw(row + 2, col - 2, " # # ");
    showit();
}

int
main(
	int argc GCC_UNUSED,
	char *argv[]GCC_UNUSED)
{
    int start, end, row, diff, flag = 0, direction;
    unsigned seed;

    CATCHALL(onsig);

    initscr();
    noecho();
    cbreak();
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);

    if (has_colors()) {
	start_color();
#if HAVE_USE_DEFAULT_COLORS
	if (use_default_colors() == OK)
	    my_bg = -1;
#endif
    }
    curs_set(0);

    seed = (unsigned) time((time_t *) 0);
    srand(seed);
    for (;;) {
	do {
	    start = rand() % (COLS - 3);
	    end = rand() % (COLS - 3);
	    start = (start < 2) ? 2 : start;
	    end = (end < 2) ? 2 : end;
	    direction = (start > end) ? -1 : 1;
	    diff = abs(start - end);
	} while (diff < 2 || diff >= LINES - 2);
	(void) attrset(AttrArg(0, A_NORMAL));
	for (row = 0; row < diff; row++) {
	    MvPrintw(LINES - row, start + (row * direction),
		     (direction < 0) ? "\\" : "/");
	    if (flag++) {
		showit();
		erase();
		flag = 0;
	    }
	}
	if (flag++) {
	    showit();
	    flag = 0;
	}
	seed = (unsigned) time((time_t *) 0);
	srand(seed);
	explode(LINES - row, start + (diff * direction));
	erase();
	showit();
    }
}
