/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
 * This is an example written by Alexander V. Lukyanov <lav@yars.free.net>,
 * to demonstrate an inconsistency between ncurses and SVr4 curses.
 *
 * $Id: testaddch.c,v 1.12 2014/08/02 23:11:20 tom Exp $
 */
#include <test.priv.h>

static void
attr_addstr(const char *s, chtype a)
{
    while (*s)
	addch(((unsigned char) (*s++)) | a);
}

int
main(
	int argc GCC_UNUSED,
	char *argv[]GCC_UNUSED)
{
    unsigned i;
    chtype back, set, attr;

    setlocale(LC_ALL, "");

    initscr();
    start_color();
    init_pair(1, COLOR_WHITE, COLOR_BLUE);
    init_pair(2, COLOR_WHITE, COLOR_RED);
    init_pair(3, COLOR_BLACK, COLOR_MAGENTA);
    init_pair(4, COLOR_BLACK, COLOR_GREEN);
    init_pair(5, COLOR_BLACK, COLOR_CYAN);
    init_pair(6, COLOR_BLACK, COLOR_YELLOW);
    init_pair(7, COLOR_BLACK, COLOR_WHITE);

    for (i = 0; i < 8; i++) {
	back = (i & 1) ? A_BOLD | 'B' : ' ';
	set = (i & 2) ? A_REVERSE : 0;
	attr = (chtype) ((i & 4) ? COLOR_PAIR(4) : 0);

	bkgdset(back);
	(void) attrset(AttrArg(set, 0));

	attr_addstr("Test string with spaces ->   <-\n", attr);
    }
    addch('\n');
    for (i = 0; i < 8; i++) {
	back = (i & 1) ? (A_BOLD | 'B' | (chtype) COLOR_PAIR(1)) : ' ';
	set = (i & 2) ? (A_REVERSE | (chtype) COLOR_PAIR(2)) : 0;
	attr = (chtype) ((i & 4) ? (chtype) COLOR_PAIR(4) : 0);

	bkgdset(back);
	(void) attrset(AttrArg(set, 0));

	attr_addstr("Test string with spaces ->   <-\n", attr);
    }

    getch();
    endwin();
    ExitProgram(EXIT_SUCCESS);
}
