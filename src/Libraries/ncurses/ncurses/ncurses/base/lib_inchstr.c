/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
**	lib_inchstr.c
**
**	The routine winchnstr().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_inchstr.c,v 1.12 2010/12/20 01:37:41 tom Exp $")

NCURSES_EXPORT(int)
winchnstr(WINDOW *win, chtype *str, int n)
{
    int i = 0;

    T((T_CALLED("winchnstr(%p,%p,%d)"), (void *) win, (void *) str, n));

    if (!str)
	returnCode(0);

    if (win) {
	for (; (n < 0 || (i < n)) && (win->_curx + i <= win->_maxx); i++)
	    str[i] =
		(chtype) CharOf(win->_line[win->_cury].text[win->_curx + i]) |
		AttrOf(win->_line[win->_cury].text[win->_curx + i]);
    }
    str[i] = (chtype) 0;

    returnCode(i);
}
