/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
**	lib_delch.c
**
**	The routine wdelch().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_delch.c,v 1.13 2009/10/24 22:32:47 tom Exp $")

NCURSES_EXPORT(int)
wdelch(WINDOW *win)
{
    int code = ERR;

    T((T_CALLED("wdelch(%p)"), (void *) win));

    if (win) {
	NCURSES_CH_T blank = win->_nc_bkgd;
	struct ldat *line = &(win->_line[win->_cury]);
	NCURSES_CH_T *end = &(line->text[win->_maxx]);
	NCURSES_CH_T *temp2 = &(line->text[win->_curx + 1]);
	NCURSES_CH_T *temp1 = temp2 - 1;

	CHANGED_TO_EOL(line, win->_curx, win->_maxx);
	while (temp1 < end)
	    *temp1++ = *temp2++;

	*temp1 = blank;

	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}
