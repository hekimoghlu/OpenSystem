/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
 * $Id: echochar.c,v 1.10 2014/08/09 22:35:51 tom Exp $
 *
 * Demonstrate the echochar function (compare to dots.c).
 * Thomas Dickey - 2006/11/4
 */

#include <test.priv.h>

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

static void
set_color(char *my_pairs, int fg, int bg)
{
    int pair = (fg * COLORS) + bg;
    if (!my_pairs[pair]) {
	init_pair((short) pair,
		  (short) fg,
		  (short) bg);
    }
    attron(COLOR_PAIR(pair));
}

int
main(int argc GCC_UNUSED,
     char *argv[]GCC_UNUSED)
{
    int ch, x, y, z, p;
    double r;
    double c;
    bool use_colors;
    bool opt_r = FALSE;
    char *my_pairs = 0;
    int last_fg = 0;
    int last_bg = 0;

    while ((ch = getopt(argc, argv, "r")) != -1) {
	switch (ch) {
	case 'r':
	    opt_r = TRUE;
	    break;
	default:
	    fprintf(stderr, "usage: echochar [-r]\n");
	    ExitProgram(EXIT_FAILURE);
	}
    }

    CATCHALL(onsig);
    initscr();

    use_colors = has_colors();
    if (use_colors) {
	start_color();
	if (COLOR_PAIRS > 0) {
	    my_pairs = typeCalloc(char, (size_t) COLOR_PAIRS);
	}
	use_colors = (my_pairs != 0);
    }

    srand((unsigned) time(0));

    curs_set(0);

    r = (double) (LINES - 4);
    c = (double) (COLS - 4);
    started = time((time_t *) 0);

    while (!interrupted) {
	x = (int) (c * ranf()) + 2;
	y = (int) (r * ranf()) + 2;
	p = (ranf() > 0.9) ? '*' : ' ';

	move(y, x);
	if (use_colors > 0) {
	    z = (int) (ranf() * COLORS);
	    if (ranf() > 0.01) {
		set_color(my_pairs, z, last_bg);
		last_fg = z;
	    } else {
		set_color(my_pairs, last_fg, z);
		last_bg = z;
		napms(1);
	    }
	} else {
	    if (ranf() <= 0.01) {
		if (ranf() > 0.6)
		    attron(A_REVERSE);
		else
		    attroff(A_REVERSE);
		napms(1);
	    }
	}
	if (opt_r) {
	    addch(UChar(p));
	    refresh();
	} else {
	    echochar(UChar(p));
	}
	++total_chars;
    }
    cleanup();
    ExitProgram(EXIT_SUCCESS);
}
