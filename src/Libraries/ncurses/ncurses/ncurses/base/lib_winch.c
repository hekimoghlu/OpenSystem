/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
 *  Author: Thomas E. Dickey <dickey@clark.net> 1998                        *
 ****************************************************************************/

/*
**	lib_winch.c
**
**	The routine winch().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_winch.c,v 1.8 2010/12/19 01:22:58 tom Exp $")

NCURSES_EXPORT(chtype)
winch(WINDOW *win)
{
    T((T_CALLED("winch(%p)"), (void *) win));
    if (win != 0) {
	returnChtype((chtype) CharOf(win->_line[win->_cury].text[win->_curx])
		     | AttrOf(win->_line[win->_cury].text[win->_curx]));
    } else {
	returnChtype(0);
    }
}
