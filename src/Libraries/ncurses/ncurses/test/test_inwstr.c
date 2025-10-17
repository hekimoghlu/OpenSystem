/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
 * $Id: test_inwstr.c,v 1.4 2010/05/01 19:13:46 tom Exp $
 *
 * Author: Thomas E Dickey
 *
 * Demonstrate the inwstr functions from the curses library.

       int inwstr(wchar_t *str);
       int innwstr(wchar_t *str, int n);
       int winwstr(WINDOW *win, wchar_t *str);
       int winnwstr(WINDOW *win, wchar_t *str, int n);
       int mvinwstr(int y, int x, wchar_t *str);
       int mvinnwstr(int y, int x, wchar_t *str, int n);
       int mvwinwstr(WINDOW *win, int y, int x, wchar_t *str);
       int mvwinnwstr(WINDOW *win, int y, int x, wchar_t *str, int n);
 */

#include <test.priv.h>

#if USE_WIDEC_SUPPORT

#define BASE_Y 6
#define MAX_COLS 1024

static bool
Quit(int ch)
{
    return (ch == ERR || ch == 'q' || ch == QUIT || ch == ESCAPE);
}

static void
show_1st(WINDOW *win, int line, wchar_t *buffer)
{
    (void) mvwaddwstr(win, line, 5, buffer);
}

static void
showmore(WINDOW *win, int line, wchar_t *buffer)
{
    wmove(win, line, 0);
    wclrtoeol(win);
    show_1st(win, line, buffer);
}

static int
test_inchs(int level, char **argv, WINDOW *chrwin, WINDOW *strwin)
{
    WINDOW *txtbox = 0;
    WINDOW *txtwin = 0;
    FILE *fp;
    int ch;
    int txt_x = 0, txt_y = 0;
    int base_y;
    int limit = getmaxx(strwin) - 5;
    wchar_t buffer[MAX_COLS];

    if (argv[level] == 0) {
	beep();
	return FALSE;
    }

    if (level > 1) {
	txtbox = newwin(LINES - BASE_Y, COLS - level, BASE_Y, level);
	box(txtbox, 0, 0);
	wnoutrefresh(txtbox);

	txtwin = derwin(txtbox,
			getmaxy(txtbox) - 2,
			getmaxx(txtbox) - 2,
			1, 1);
	base_y = 0;
    } else {
	txtwin = stdscr;
	base_y = BASE_Y;
    }

    keypad(txtwin, TRUE);	/* enable keyboard mapping */
    (void) cbreak();		/* take input chars one at a time, no wait for \n */
    (void) noecho();		/* don't echo input */

    txt_y = base_y;
    txt_x = 0;
    wmove(txtwin, txt_y, txt_x);

    if ((fp = fopen(argv[level], "r")) != 0) {
	while ((ch = fgetc(fp)) != EOF) {
	    if (waddch(txtwin, UChar(ch)) != OK) {
		break;
	    }
	}
	fclose(fp);
    } else {
	wprintw(txtwin, "Cannot open:\n%s", argv[1]);
    }

    while (!Quit(ch = mvwgetch(txtwin, txt_y, txt_x))) {
	switch (ch) {
	case KEY_DOWN:
	case 'j':
	    if (txt_y < getmaxy(txtwin) - 1)
		txt_y++;
	    else
		beep();
	    break;
	case KEY_UP:
	case 'k':
	    if (txt_y > base_y)
		txt_y--;
	    else
		beep();
	    break;
	case KEY_LEFT:
	case 'h':
	    if (txt_x > 0)
		txt_x--;
	    else
		beep();
	    break;
	case KEY_RIGHT:
	case 'l':
	    if (txt_x < getmaxx(txtwin) - 1)
		txt_x++;
	    else
		beep();
	    break;
	case 'w':
	    test_inchs(level + 1, argv, chrwin, strwin);
	    if (txtbox != 0) {
		touchwin(txtbox);
		wnoutrefresh(txtbox);
	    } else {
		touchwin(txtwin);
		wnoutrefresh(txtwin);
	    }
	    break;
	case '-':
	    if (limit > 0) {
		--limit;
	    } else {
		beep();
	    }
	    break;
	case '+':
	    ++limit;
	    break;
	default:
	    beep();
	    break;
	}

	MvWPrintw(chrwin, 0, 0, "line:");
	wclrtoeol(chrwin);

	if (txtwin != stdscr) {
	    wmove(txtwin, txt_y, txt_x);

	    if (winwstr(txtwin, buffer) != ERR) {
		show_1st(chrwin, 0, buffer);
	    }
	    if (mvwinwstr(txtwin, txt_y, txt_x, buffer) != ERR) {
		showmore(chrwin, 1, buffer);
	    }
	} else {
	    move(txt_y, txt_x);

	    if (inwstr(buffer) != ERR) {
		show_1st(chrwin, 0, buffer);
	    }
	    if (mvinwstr(txt_y, txt_x, buffer) != ERR) {
		showmore(chrwin, 1, buffer);
	    }
	}
	wnoutrefresh(chrwin);

	MvWPrintw(strwin, 0, 0, "%4d:", limit);
	wclrtobot(strwin);

	if (txtwin != stdscr) {
	    wmove(txtwin, txt_y, txt_x);
	    if (winnwstr(txtwin, buffer, limit) != ERR) {
		show_1st(strwin, 0, buffer);
	    }

	    if (mvwinnwstr(txtwin, txt_y, txt_x, buffer, limit) != ERR) {
		showmore(strwin, 1, buffer);
	    }
	} else {
	    move(txt_y, txt_x);
	    if (innwstr(buffer, limit) != ERR) {
		show_1st(strwin, 0, buffer);
	    }

	    if (mvinnwstr(txt_y, txt_x, buffer, limit) != ERR) {
		showmore(strwin, 1, buffer);
	    }
	}

	wnoutrefresh(strwin);
    }
    if (level > 1) {
	delwin(txtwin);
	delwin(txtbox);
    }
    return TRUE;
}

int
main(int argc, char *argv[])
{
    WINDOW *chrbox;
    WINDOW *chrwin;
    WINDOW *strwin;

    setlocale(LC_ALL, "");

    if (argc < 2) {
	fprintf(stderr, "usage: %s file\n", argv[0]);
	return EXIT_FAILURE;
    }

    initscr();

    chrbox = derwin(stdscr, BASE_Y, COLS, 0, 0);
    box(chrbox, 0, 0);
    wnoutrefresh(chrbox);

    chrwin = derwin(chrbox, 2, COLS - 2, 1, 1);
    strwin = derwin(chrbox, 2, COLS - 2, 3, 1);

    test_inchs(1, argv, chrwin, strwin);

    endwin();
    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(void)
{
    printf("This program requires the wide-ncurses library\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
