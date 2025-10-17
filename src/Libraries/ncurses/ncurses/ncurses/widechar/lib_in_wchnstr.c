/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
**	lib_in_wchnstr.c
**
**	The routine win_wchnstr().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_in_wchnstr.c,v 1.8 2009/10/24 22:37:48 tom Exp $")

NCURSES_EXPORT(int)
win_wchnstr(WINDOW *win, cchar_t *wchstr, int n)
{
    int code = OK;

    T((T_CALLED("win_wchnstr(%p,%p,%d)"), (void *) win, (void *) wchstr, n));
    if (win != 0
	&& wchstr != 0) {
	NCURSES_CH_T *src;
	int row, col;
	int j, k, limit;

	getyx(win, row, col);
	limit = getmaxx(win) - col;
	src = &(win->_line[row].text[col]);

	if (n < 0) {
	    n = limit;
	} else if (n > limit) {
	    n = limit;
	}
	for (j = k = 0; j < n; ++j) {
	    if (j == 0 || !WidecExt(src[j]) || isWidecBase(src[j])) {
		wchstr[k++] = src[j];
	    }
	}
	memset(&(wchstr[k]), 0, sizeof(*wchstr));
	T(("result = %s", _nc_viscbuf(wchstr, n)));
    } else {
	code = ERR;
    }
    returnCode(code);
}
