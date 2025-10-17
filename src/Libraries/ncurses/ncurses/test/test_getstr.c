/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
 * $Id: test_getstr.c,v 1.10 2012/07/07 18:22:49 tom Exp $
 *
 * Author: Thomas E Dickey
 *
 * Demonstrate the getstr functions from the curses library.

       int getstr(char *str);
       int getnstr(char *str, int n);
       int wgetstr(WINDOW *win, char *str);
       int wgetnstr(WINDOW *win, char *str, int n);
       int mvgetstr(int y, int x, char *str);
       int mvwgetstr(WINDOW *win, int y, int x, char *str);
       int mvgetnstr(int y, int x, char *str, int n);
       int mvwgetnstr(WINDOW *, int y, int x, char *str, int n);
 */

#include <test.priv.h>

#if HAVE_CHGAT
/* Solaris SVr4 curses lacks wchgat, mvgetnstr, mvwgetnstr */

#define BASE_Y 6
#define MAX_COLS 1024

typedef enum {
    eGetStr = 0,
    eGetNStr,
    eMvGetStr,
    eMvGetNStr,
    eMaxFlavor
} Flavors;

/*
 * Return-code is OK/ERR or a keyname.
 */
static const char *
ok_keyname(int code)
{
    return ((code == OK) ? "OK" : ((code == ERR) ? "ERR" : keyname(code)));
}

static bool
Quit(int ch)
{
    return (ch == ERR || ch == 'q' || ch == QUIT || ch == ESCAPE);
}

static int
Remainder(WINDOW *txtwin)
{
    int result = getmaxx(txtwin) - getcurx(txtwin);
    return (result > 0) ? result : 0;
}

/*
 * Show a highlighted line in the place where input will happen.
 */
static void
ShowPrompt(WINDOW *txtwin, int limit)
{
    wchgat(txtwin, limit, A_REVERSE, 0, NULL);
    wnoutrefresh(txtwin);
}

static void
MovePrompt(WINDOW *txtwin, int limit, int y, int x)
{
    wchgat(txtwin, Remainder(txtwin), A_NORMAL, 0, NULL);
    wmove(txtwin, y, x);
    ShowPrompt(txtwin, limit);
}

static int
ShowFlavor(WINDOW *strwin, WINDOW *txtwin, int flavor, int limit)
{
    const char *name = "?";
    bool limited = FALSE;
    bool wins = (txtwin != stdscr);
    int result;

    switch (flavor) {
    case eGetStr:
	name = wins ? "wgetstr" : "getstr";
	break;
    case eGetNStr:
	limited = TRUE;
	name = wins ? "wgetnstr" : "getnstr";
	break;
    case eMvGetStr:
	name = wins ? "mvwgetstr" : "mvgetstr";
	break;
    case eMvGetNStr:
	limited = TRUE;
	name = wins ? "mvwgetnstr" : "mvgetnstr";
	break;
    case eMaxFlavor:
	break;
    }

    wmove(strwin, 0, 0);
    werase(strwin);

    if (limited) {
	wprintw(strwin, "%s(%d):", name, limit);
    } else {
	wprintw(strwin, "%s:", name);
    }
    result = limited ? limit : Remainder(txtwin);
    ShowPrompt(txtwin, result);

    wnoutrefresh(strwin);
    return result;
}

static int
test_getstr(int level, char **argv, WINDOW *strwin)
{
    WINDOW *txtbox = 0;
    WINDOW *txtwin = 0;
    FILE *fp;
    int ch;
    int rc;
    int txt_x = 0, txt_y = 0;
    int base_y;
    int flavor = 0;
    int limit = getmaxx(strwin) - 5;
    int actual;

    char buffer[MAX_COLS];

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

    wmove(txtwin, txt_y, txt_x);
    actual = ShowFlavor(strwin, txtwin, flavor, limit);
    while (!Quit(ch = mvwgetch(txtwin, txt_y, txt_x))) {
	switch (ch) {
	case KEY_DOWN:
	case 'j':
	    if (txt_y < getmaxy(txtwin) - 1) {
		MovePrompt(txtwin, actual, ++txt_y, txt_x);
	    } else {
		beep();
	    }
	    break;
	case KEY_UP:
	case 'k':
	    if (txt_y > base_y) {
		MovePrompt(txtwin, actual, --txt_y, txt_x);
	    } else {
		beep();
	    }
	    break;
	case KEY_LEFT:
	case 'h':
	    if (txt_x > 0) {
		MovePrompt(txtwin, actual, txt_y, --txt_x);
	    } else {
		beep();
	    }
	    break;
	case KEY_RIGHT:
	case 'l':
	    if (txt_x < getmaxx(txtwin) - 1) {
		MovePrompt(txtwin, actual, txt_y, ++txt_x);
	    } else {
		beep();
	    }
	    break;

	case 'w':
	    test_getstr(level + 1, argv, strwin);
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
		actual = ShowFlavor(strwin, txtwin, flavor, --limit);
		MovePrompt(txtwin, actual, txt_y, txt_x);
	    } else {
		beep();
	    }
	    break;

	case '+':
	    actual = ShowFlavor(strwin, txtwin, flavor, ++limit);
	    MovePrompt(txtwin, actual, txt_y, txt_x);
	    break;

	case '<':
	    if (flavor > 0) {
		actual = ShowFlavor(strwin, txtwin, --flavor, limit);
		MovePrompt(txtwin, actual, txt_y, txt_x);
	    } else {
		beep();
	    }
	    break;

	case '>':
	    if (flavor + 1 < eMaxFlavor) {
		actual = ShowFlavor(strwin, txtwin, ++flavor, limit);
		MovePrompt(txtwin, actual, txt_y, txt_x);
	    } else {
		beep();
	    }
	    break;

	case ':':
	    actual = ShowFlavor(strwin, txtwin, flavor, limit);
	    *buffer = '\0';
	    rc = ERR;
	    echo();
	    (void) wattrset(txtwin, A_REVERSE);
	    switch (flavor) {
	    case eGetStr:
		if (txtwin != stdscr) {
		    wmove(txtwin, txt_y, txt_x);
		    rc = wgetstr(txtwin, buffer);
		} else {
		    move(txt_y, txt_x);
		    rc = getstr(buffer);
		}
		break;
	    case eGetNStr:
		if (txtwin != stdscr) {
		    wmove(txtwin, txt_y, txt_x);
		    rc = wgetnstr(txtwin, buffer, limit);
		} else {
		    move(txt_y, txt_x);
		    rc = getnstr(buffer, limit);
		}
		break;
	    case eMvGetStr:
		if (txtwin != stdscr) {
		    rc = mvwgetstr(txtwin, txt_y, txt_x, buffer);
		} else {
		    rc = mvgetstr(txt_y, txt_x, buffer);
		}
		break;
	    case eMvGetNStr:
		if (txtwin != stdscr) {
		    rc = mvwgetnstr(txtwin, txt_y, txt_x, buffer, limit);
		} else {
		    rc = mvgetnstr(txt_y, txt_x, buffer, limit);
		}
		break;
	    case eMaxFlavor:
		break;
	    }
	    noecho();
	    (void) wattrset(txtwin, A_NORMAL);
	    wprintw(strwin, "%s:%s", ok_keyname(rc), buffer);
	    wnoutrefresh(strwin);
	    break;
	default:
	    beep();
	    break;
	}
	doupdate();
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

    strwin = derwin(chrbox, 4, COLS - 2, 1, 1);

    test_getstr(1, argv, strwin);

    endwin();
    ExitProgram(EXIT_SUCCESS);
}

#else
int
main(void)
{
    printf("This program requires the curses chgat function\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
