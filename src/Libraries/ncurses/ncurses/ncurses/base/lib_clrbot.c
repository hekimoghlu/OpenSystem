/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
**	lib_clrbot.c
**
**	The routine wclrtobot().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_clrbot.c,v 1.21 2009/10/24 22:33:19 tom Exp $")

NCURSES_EXPORT(int)
wclrtobot(WINDOW *win)
{
    int code = ERR;

    T((T_CALLED("wclrtobot(%p)"), (void *) win));

    if (win) {
	NCURSES_SIZE_T y;
	NCURSES_SIZE_T startx = win->_curx;
	NCURSES_CH_T blank = win->_nc_bkgd;

	T(("clearing from y = %ld to y = %ld with maxx =  %ld",
	   (long) win->_cury, (long) win->_maxy, (long) win->_maxx));

	for (y = win->_cury; y <= win->_maxy; y++) {
	    struct ldat *line = &(win->_line[y]);
	    NCURSES_CH_T *ptr = &(line->text[startx]);
	    NCURSES_CH_T *end = &(line->text[win->_maxx]);

	    CHANGED_TO_EOL(line, startx, win->_maxx);

	    while (ptr <= end)
		*ptr++ = blank;

	    startx = 0;
	}
	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
