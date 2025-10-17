/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
 *  Author: Thomas E. Dickey <dickey@clark.net> 1997                        *
 ****************************************************************************/

/*
 *	lib_redrawln.c
 *
 *	The routine wredrawln().
 *
 */

#include <curses.priv.h>

MODULE_ID("$Id: lib_redrawln.c,v 1.17 2010/12/19 00:03:23 tom Exp $")

NCURSES_EXPORT(int)
wredrawln(WINDOW *win, int beg, int num)
{
    int i;
    int end;
    size_t len;
    SCREEN *sp;

    T((T_CALLED("wredrawln(%p,%d,%d)"), (void *) win, beg, num));

    if (win == 0)
	returnCode(ERR);

    sp = _nc_screen_of(win);

    if (beg < 0)
	beg = 0;

    if (touchline(win, beg, num) == ERR)
	returnCode(ERR);

    if (touchline(CurScreen(sp), beg + win->_begy, num) == ERR)
	returnCode(ERR);

    end = beg + num;
    if (end > CurScreen(sp)->_maxy + 1 - win->_begy)
	end = CurScreen(sp)->_maxy + 1 - win->_begy;
    if (end > win->_maxy + 1)
	end = win->_maxy + 1;

    len = (size_t) (win->_maxx + 1);
    if (len > (size_t) (CurScreen(sp)->_maxx + 1 - win->_begx))
	len = (size_t) (CurScreen(sp)->_maxx + 1 - win->_begx);
    len *= sizeof(CurScreen(sp)->_line[0].text[0]);

    for (i = beg; i < end; i++) {
	int crow = i + win->_begy;

	memset(CurScreen(sp)->_line[crow].text + win->_begx, 0, len);
	NCURSES_SP_NAME(_nc_make_oldhash) (NCURSES_SP_ARGx crow);
    }

    returnCode(OK);
}
