/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
**	lib_vline_set.c
**
**	The routine wvline_set().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_vline_set.c,v 1.4 2010/12/19 01:50:50 tom Exp $")

NCURSES_EXPORT(int)
wvline_set(WINDOW *win, const cchar_t *ch, int n)
{
    int code = ERR;
    int row, col;
    int end;

    T((T_CALLED("wvline(%p,%s,%d)"), (void *) win, _tracecchar_t(ch), n));

    if (win) {
	NCURSES_CH_T wch;
	row = win->_cury;
	col = win->_curx;
	end = row + n - 1;
	if (end > win->_maxy)
	    end = win->_maxy;

	if (ch == 0)
	    wch = *WACS_VLINE;
	else
	    wch = *ch;
	wch = _nc_render(win, wch);

	while (end >= row) {
	    struct ldat *line = &(win->_line[end]);
	    line->text[col] = wch;
	    CHANGED_CELL(line, col);
	    end--;
	}

	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
