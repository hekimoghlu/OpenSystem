/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

/*
**	lib_erase.c
**
**	The routine werase().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_erase.c,v 1.17 2009/10/24 22:32:29 tom Exp $")

NCURSES_EXPORT(int)
werase(WINDOW *win)
{
    int code = ERR;
    int y;
    NCURSES_CH_T blank;
    NCURSES_CH_T *sp, *end, *start;

    T((T_CALLED("werase(%p)"), (void *) win));

    if (win) {
	blank = win->_nc_bkgd;
	for (y = 0; y <= win->_maxy; y++) {
	    start = win->_line[y].text;
	    end = &start[win->_maxx];

	    /*
	     * If this is a derived window, we have to handle the case where
	     * a multicolumn character extends into the window that we are
	     * erasing.
	     */
	    if_WIDEC({
		if (isWidecExt(start[0])) {
		    int x = (win->_parent != 0) ? (win->_begx) : 0;
		    while (x-- > 0) {
			if (isWidecBase(start[-1])) {
			    --start;
			    break;
			}
			--start;
		    }
		}
	    });

	    for (sp = start; sp <= end; sp++)
		*sp = blank;

	    win->_line[y].firstchar = 0;
	    win->_line[y].lastchar = win->_maxx;
	}
	win->_curx = win->_cury = 0;
	win->_flags &= ~_WRAPPED;
	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
