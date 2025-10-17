/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
**	lib_hline_set.c
**
**	The routine whline_set().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_hline_set.c,v 1.4 2010/12/19 01:45:03 tom Exp $")

NCURSES_EXPORT(int)
whline_set(WINDOW *win, const cchar_t *ch, int n)
{
    int code = ERR;
    int start;
    int end;

    T((T_CALLED("whline_set(%p,%s,%d)"), (void *) win, _tracecchar_t(ch), n));

    if (win) {
	struct ldat *line = &(win->_line[win->_cury]);
	NCURSES_CH_T wch;

	start = win->_curx;
	end = start + n - 1;
	if (end > win->_maxx)
	    end = win->_maxx;

	CHANGED_RANGE(line, start, end);

	if (ch == 0)
	    wch = *WACS_HLINE;
	else
	    wch = *ch;
	wch = _nc_render(win, wch);

	while (end >= start) {
	    line->text[end] = wch;
	    end--;
	}

	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
