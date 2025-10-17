/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
**	lib_insnstr.c
**
**	The routine winsnstr().
**
*/

#include <curses.priv.h>
#include <ctype.h>

MODULE_ID("$Id: lib_insnstr.c,v 1.3 2009/10/24 22:04:35 tom Exp $")

NCURSES_EXPORT(int)
winsnstr(WINDOW *win, const char *s, int n)
{
    int code = ERR;
    NCURSES_SIZE_T oy;
    NCURSES_SIZE_T ox;
    const unsigned char *str = (const unsigned char *) s;
    const unsigned char *cp;

    T((T_CALLED("winsnstr(%p,%s,%d)"), (void *) win, _nc_visbufn(s, n), n));

    if (win != 0 && str != 0) {
	SCREEN *sp = _nc_screen_of(win);

	oy = win->_cury;
	ox = win->_curx;
	for (cp = str; *cp && (n <= 0 || (cp - str) < n); cp++) {
	    _nc_insert_ch(sp, win, (chtype) UChar(*cp));
	}
	win->_curx = ox;
	win->_cury = oy;
	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
