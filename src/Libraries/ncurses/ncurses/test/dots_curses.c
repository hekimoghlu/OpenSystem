/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
 * Author: Thomas E. Dickey
 *
 * $Id: dots_curses.c,v 1.3 2014/08/09 22:28:42 tom Exp $
 *
 * A simple demo of the curses interface used for comparison with termcap.
 */
#include <test.priv.h>

#if !defined(__MINGW32__)
#include <sys/time.h>
#endif

#include <time.h>

#define valid(s) ((s != 0) && s != (char *)-1)

static bool interrupted = FALSE;
static long total_chars = 0;
static time_t started;

static void
cleanup(void)
{
    endwin();

    printf("\n\n%ld total chars, rate %.2f/sec\n",
	   total_chars,
	   ((double) (total_chars) / (double) (time((time_t *) 0) - started)));
}

static void
onsig(int n GCC_UNUSED)
{
    interrupted = TRUE;
}

static double
ranf(void)
{
    long r = (rand() & 077777);
    return ((double) r / 32768.);
}

static int
mypair(int fg, int bg)
{
    int pair = (fg * COLORS) + bg;
    return (pair >= COLOR_PAIRS) ? -1 : pair;
}

static void
set_colors(int fg, int bg)
{
    int pair = mypair(fg, bg);
    if (pair > 0) {
	attron(COLOR_PAIR(mypair(fg, bg)));
    }
}

int
main(int argc GCC_UNUSED,
     char *argv[]GCC_UNUSED)
{
    int x, y, z, p;
    int fg, bg;
    double r;
    double c;

    CATCHALL(onsig);

    srand((unsigned) time(0));

    initscr();
    if (has_colors()) {
	start_color();
	for (fg = 0; fg < COLORS; fg++) {
	    for (bg = 0; bg < COLORS; bg++) {
		int pair = mypair(fg, bg);
		if (pair > 0)
		    init_pair((short) pair, (short) fg, (short) bg);
	    }
	}
    }

    r = (double) (LINES - 4);
    c = (double) (COLS - 4);
    started = time((time_t *) 0);

    fg = COLOR_WHITE;
    bg = COLOR_BLACK;
    while (!interrupted) {
	x = (int) (c * ranf()) + 2;
	y = (int) (r * ranf()) + 2;
	p = (ranf() > 0.9) ? '*' : ' ';

	move(y, x);
	if (has_colors()) {
	    z = (int) (ranf() * COLORS);
	    if (ranf() > 0.01) {
		set_colors(fg = z, bg);
		attron(COLOR_PAIR(mypair(fg, bg)));
	    } else {
		set_colors(fg, bg = z);
		napms(1);
	    }
	} else {
	    if (ranf() <= 0.01) {
		if (ranf() > 0.6) {
		    attron(A_REVERSE);
		} else {
		    attroff(A_REVERSE);
		}
		napms(1);
	    }
	}
	addch((chtype) p);
	refresh();
	++total_chars;
    }
    cleanup();
    ExitProgram(EXIT_SUCCESS);
}
