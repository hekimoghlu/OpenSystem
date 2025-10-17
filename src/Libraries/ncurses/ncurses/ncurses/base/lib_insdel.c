/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
**	lib_insdel.c
**
**	The routine winsdelln(win, n).
**  positive n insert n lines above current line
**  negative n delete n lines starting from current line
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_insdel.c,v 1.13 2009/10/24 22:34:41 tom Exp $")

NCURSES_EXPORT(int)
winsdelln(WINDOW *win, int n)
{
    int code = ERR;

    T((T_CALLED("winsdelln(%p,%d)"), (void *) win, n));

    if (win) {
	if (n != 0) {
	    _nc_scroll_window(win, -n, win->_cury, win->_maxy,
			      win->_nc_bkgd);
	    _nc_synchook(win);
	}
	code = OK;
    }
    returnCode(code);
}
