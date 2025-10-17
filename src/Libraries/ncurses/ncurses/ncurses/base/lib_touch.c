/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
**	lib_touch.c
**
**	   The routines	untouchwin(),
**			wtouchln(),
**			is_linetouched()
**			is_wintouched().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_touch.c,v 1.12 2012/06/09 20:29:33 tom Exp $")

NCURSES_EXPORT(bool)
is_linetouched(WINDOW *win, int line)
{
    T((T_CALLED("is_linetouched(%p,%d)"), (void *) win, line));

    /* XSI doesn't define any error */
    if (!win || (line > win->_maxy) || (line < 0))
	returnCode((bool) ERR);

    returnCode(win->_line[line].firstchar != _NOCHANGE ? TRUE : FALSE);
}

NCURSES_EXPORT(bool)
is_wintouched(WINDOW *win)
{
    int i;

    T((T_CALLED("is_wintouched(%p)"), (void *) win));

    if (win)
	for (i = 0; i <= win->_maxy; i++)
	    if (win->_line[i].firstchar != _NOCHANGE)
		returnCode(TRUE);
    returnCode(FALSE);
}

NCURSES_EXPORT(int)
wtouchln(WINDOW *win, int y, int n, int changed)
{
    int i;

    T((T_CALLED("wtouchln(%p,%d,%d,%d)"), (void *) win, y, n, changed));

    if (!win || (n < 0) || (y < 0) || (y > win->_maxy))
	returnCode(ERR);

    for (i = y; i < y + n; i++) {
	if (i > win->_maxy)
	    break;
	win->_line[i].firstchar = (NCURSES_SIZE_T) (changed ? 0 : _NOCHANGE);
	win->_line[i].lastchar = (NCURSES_SIZE_T) (changed
						   ? win->_maxx
						   : _NOCHANGE);
    }
    returnCode(OK);
}
