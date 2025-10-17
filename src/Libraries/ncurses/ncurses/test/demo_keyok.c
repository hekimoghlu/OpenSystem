/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
 * $Id: demo_keyok.c,v 1.5 2006/11/04 20:09:51 tom Exp $
 *
 * Demonstrate the keyok() function.
 * Thomas Dickey - 2002/11/23
 */

#include <test.priv.h>

#if defined(NCURSES_VERSION) && NCURSES_EXT_FUNCS
int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    int lastch = ERR;
    int ch;
    WINDOW *win;

    initscr();
    (void) cbreak();		/* take input chars one at a time, no wait for \n */
    (void) noecho();		/* don't echo input */

    printw("Typing any function key will disable it, but typing it twice in\n");
    printw("a row will turn it back on (just for a demo).");
    refresh();

    win = newwin(LINES - 2, COLS, 2, 0);
    scrollok(win, TRUE);
    keypad(win, TRUE);
    wmove(win, 0, 0);

    while ((ch = wgetch(win)) != ERR) {
	const char *name = keyname(ch);
	wprintw(win, "Keycode %d, name %s\n",
		ch,
		name != 0 ? name : "<null>");
	wclrtoeol(win);
	wrefresh(win);
	if (ch >= KEY_MIN) {
	    keyok(ch, FALSE);
	    lastch = ch;
	} else if (lastch >= KEY_MIN) {
	    keyok(lastch, TRUE);
	}
    }
    endwin();
    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(void)
{
    printf("This program requires the ncurses library\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
