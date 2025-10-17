/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 ****************************************************************************/

/*
**	lib_clreol.c
**
**	The routine wclrtoeol().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_clreol.c,v 1.22 2009/10/24 22:33:06 tom Exp $")

NCURSES_EXPORT(int)
wclrtoeol(WINDOW *win)
{
    int code = ERR;

    T((T_CALLED("wclrtoeol(%p)"), (void *) win));

    if (win) {
	NCURSES_CH_T blank;
	NCURSES_CH_T *ptr, *end;
	struct ldat *line;
	NCURSES_SIZE_T y = win->_cury;
	NCURSES_SIZE_T x = win->_curx;

	/*
	 * If we have just wrapped the cursor, the clear applies to the
	 * new line, unless we are at the lower right corner.
	 */
	if ((win->_flags & _WRAPPED) != 0
	    && y < win->_maxy) {
	    win->_flags &= ~_WRAPPED;
	}

	/*
	 * There's no point in clearing if we're not on a legal
	 * position, either.
	 */
	if ((win->_flags & _WRAPPED) != 0
	    || y > win->_maxy
	    || x > win->_maxx)
	    returnCode(ERR);

	blank = win->_nc_bkgd;
	line = &win->_line[y];
	CHANGED_TO_EOL(line, x, win->_maxx);

	ptr = &(line->text[x]);
	end = &(line->text[win->_maxx]);

	while (ptr <= end)
	    *ptr++ = blank;

	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
