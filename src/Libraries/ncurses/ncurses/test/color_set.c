/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
 * $Id: color_set.c,v 1.8 2014/02/01 22:10:42 tom Exp $
 */

#include <test.priv.h>

#if HAVE_COLOR_SET

#define SHOW(n) ((n) == ERR ? "ERR" : "OK")

int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    NCURSES_COLOR_T f, b;
    int i;

    initscr();
    cbreak();
    noecho();

    if (has_colors()) {
	start_color();

	(void) pair_content(0, &f, &b);
	printw("pair 0 contains (%d,%d)\n", (int) f, (int) b);
	getch();

	printw("Initializing pair 1 to red/black\n");
	init_pair(1, COLOR_RED, COLOR_BLACK);
	i = color_set(1, NULL);
	printw("RED/BLACK (%s)\n", SHOW(i));
	getch();

	printw("Initializing pair 2 to white/blue\n");
	init_pair(2, COLOR_WHITE, COLOR_BLUE);
	i = color_set(2, NULL);
	printw("WHITE/BLUE (%s)\n", SHOW(i));
	getch();

	printw("Resetting colors to pair 0\n");
	i = color_set(0, NULL);
	printw("Default Colors (%s)\n", SHOW(i));
	getch();

	printw("Resetting colors to pair 1\n");
	i = color_set(1, NULL);
	printw("RED/BLACK (%s)\n", SHOW(i));
	getch();

    } else {
	printw("This demo requires a color terminal");
	getch();
    }
    endwin();

    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(void)
{
    printf("This program requires the curses color_set function\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
