/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
**	lib_move.c
**
**	The routine wmove().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_move.c,v 1.13 2009/10/24 22:34:07 tom Exp $")

NCURSES_EXPORT(int)
wmove(WINDOW *win, int y, int x)
{
    T((T_CALLED("wmove(%p,%d,%d)"), (void *) win, y, x));

    if (LEGALYX(win, y, x)) {
	win->_curx = (NCURSES_SIZE_T) x;
	win->_cury = (NCURSES_SIZE_T) y;

	win->_flags &= ~_WRAPPED;
	win->_flags |= _HASMOVED;
	returnCode(OK);
    } else
	returnCode(ERR);
}
