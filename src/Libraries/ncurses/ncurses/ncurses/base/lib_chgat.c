/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
 *     and: Sven Verdoolaege                        2001                    *
 *     and: Thomas E. Dickey                        2005                    *
 ****************************************************************************/

/*
**	lib_chgat.c
**
**	The routine wchgat().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_chgat.c,v 1.10 2014/02/01 22:13:31 tom Exp $")

NCURSES_EXPORT(int)
wchgat(WINDOW *win,
       int n,
       attr_t attr,
       NCURSES_PAIRS_T color,
       const void *opts GCC_UNUSED)
{
    int i;

    T((T_CALLED("wchgat(%p,%d,%s,%d)"),
       (void *) win,
       n,
       _traceattr(attr),
       (int) color));

    if (win) {
	struct ldat *line = &(win->_line[win->_cury]);

	toggle_attr_on(attr, ColorPair(color));

	for (i = win->_curx; i <= win->_maxx && (n == -1 || (n-- > 0)); i++) {
	    SetAttr(line->text[i], attr);
	    SetPair(line->text[i], color);
	    CHANGED_CELL(line, i);
	}

	returnCode(OK);
    } else
	returnCode(ERR);
}
