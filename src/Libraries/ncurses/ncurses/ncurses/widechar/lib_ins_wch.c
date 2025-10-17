/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
 *  Author: Thomas Dickey 2002                                              *
 ****************************************************************************/

/*
**	lib_ins_wch.c
**
**	The routine wins_wch().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_ins_wch.c,v 1.17 2011/10/22 16:34:50 tom Exp $")

/*
 * Insert the given character, updating the current location to simplify
 * inserting a string.
 */
NCURSES_EXPORT(int)
_nc_insert_wch(WINDOW *win, const cchar_t *wch)
{
    int cells = wcwidth(CharOf(CHDEREF(wch)));
    int cell;
    int code = OK;

    if (cells < 0) {
	code = winsch(win, (chtype) CharOf(CHDEREF(wch)));
    } else {
	if (cells == 0)
	    cells = 1;

	if (win->_curx <= win->_maxx) {
	    struct ldat *line = &(win->_line[win->_cury]);
	    NCURSES_CH_T *end = &(line->text[win->_curx]);
	    NCURSES_CH_T *temp1 = &(line->text[win->_maxx]);
	    NCURSES_CH_T *temp2 = temp1 - cells;

	    CHANGED_TO_EOL(line, win->_curx, win->_maxx);
	    while (temp1 > end)
		*temp1-- = *temp2--;

	    *temp1 = _nc_render(win, *wch);
	    for (cell = 1; cell < cells; ++cell) {
		SetWidecExt(temp1[cell], cell);
	    }

	    win->_curx++;
	}
    }
    return code;
}

NCURSES_EXPORT(int)
wins_wch(WINDOW *win, const cchar_t *wch)
{
    NCURSES_SIZE_T oy;
    NCURSES_SIZE_T ox;
    int code = ERR;

    T((T_CALLED("wins_wch(%p, %s)"), (void *) win, _tracecchar_t(wch)));

    if (win != 0) {
	oy = win->_cury;
	ox = win->_curx;

	code = _nc_insert_wch(win, wch);

	win->_curx = ox;
	win->_cury = oy;
	_nc_synchook(win);
    }
    returnCode(code);
}

NCURSES_EXPORT(int)
wins_nwstr(WINDOW *win, const wchar_t *wstr, int n)
{
    int code = ERR;
    NCURSES_SIZE_T oy;
    NCURSES_SIZE_T ox;
    const wchar_t *cp;

    T((T_CALLED("wins_nwstr(%p,%s,%d)"),
       (void *) win, _nc_viswbufn(wstr, n), n));

    if (win != 0
	&& wstr != 0) {
	if (n < 1)
	    n = (int) wcslen(wstr);
	code = OK;
	if (n > 0) {
	    SCREEN *sp = _nc_screen_of(win);

	    oy = win->_cury;
	    ox = win->_curx;
	    for (cp = wstr; *cp && ((cp - wstr) < n); cp++) {
		int len = wcwidth(*cp);

		if ((len >= 0 && len != 1) || !is7bits(*cp)) {
		    cchar_t tmp_cchar;
		    wchar_t tmp_wchar = *cp;
		    memset(&tmp_cchar, 0, sizeof(tmp_cchar));
		    (void) setcchar(&tmp_cchar,
				    &tmp_wchar,
				    WA_NORMAL,
				    (short) 0,
				    (void *) 0);
		    code = _nc_insert_wch(win, &tmp_cchar);
		} else {
		    /* tabs, other ASCII stuff */
		    code = _nc_insert_ch(sp, win, (chtype) (*cp));
		}
		if (code != OK)
		    break;
	    }

	    win->_curx = ox;
	    win->_cury = oy;
	    _nc_synchook(win);
	}
    }
    returnCode(code);
}
