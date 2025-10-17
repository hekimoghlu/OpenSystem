/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
/****************************************************************************
 * Author: Thomas Dickey                                                    *
 ****************************************************************************/

/*
**	lib_inwstr.c
**
**	The routines winnwstr() and winwstr().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_inwstr.c,v 1.6 2011/05/28 22:49:49 tom Exp $")

NCURSES_EXPORT(int)
winnwstr(WINDOW *win, wchar_t *wstr, int n)
{
    int row, col, inx;
    int count = 0;
    int last = 0;
    cchar_t *text;
    wchar_t wch;

    T((T_CALLED("winnwstr(%p,%p,%d)"), (void *) win, (void *) wstr, n));
    if (wstr != 0) {
	if (win) {
	    getyx(win, row, col);

	    text = win->_line[row].text;
	    while (count < n && count != ERR) {
		if (!isWidecExt(text[col])) {
		    for (inx = 0; (inx < CCHARW_MAX)
			 && ((wch = text[col].chars[inx]) != 0);
			 ++inx) {
			if (count + 1 > n) {
			    if ((count = last) == 0) {
				count = ERR;	/* error if we store nothing */
			    }
			    break;
			}
			wstr[count++] = wch;
		    }
		}
		last = count;
		if (++col > win->_maxx) {
		    break;
		}
	    }
	}
	if (count > 0) {
	    wstr[count] = '\0';
	    T(("winnwstr returns %s", _nc_viswbuf(wstr)));
	}
    }
    returnCode(count);
}

/*
 * X/Open says winwstr() returns OK if not ERR.  If that is not a blunder, it
 * must have a null termination on the string (see above).  Unlike winnstr(),
 * it does not define what happens for a negative count with winnwstr().
 */
NCURSES_EXPORT(int)
winwstr(WINDOW *win, wchar_t *wstr)
{
    int result = OK;

    T((T_CALLED("winwstr(%p,%p)"), (void *) win, (void *) wstr));
    if (win == 0) {
	result = ERR;
    } else if (winnwstr(win, wstr,
			CCHARW_MAX * (win->_maxx - win->_curx + 1)) == ERR) {
	result = ERR;
    }
    returnCode(result);
}
