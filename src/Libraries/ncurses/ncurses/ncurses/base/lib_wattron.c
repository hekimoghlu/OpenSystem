/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
**	lib_wattron.c
**
**	The routines wattr_on().
**
*/

#include <curses.priv.h>
#include <ctype.h>

MODULE_ID("$Id: lib_wattron.c,v 1.11 2010/03/31 23:38:02 tom Exp $")

NCURSES_EXPORT(int)
wattr_on(WINDOW *win, attr_t at, void *opts GCC_UNUSED)
{
    T((T_CALLED("wattr_on(%p,%s)"), (void *) win, _traceattr(at)));
    if (win != 0) {
	T(("... current %s (%d)",
	   _traceattr(WINDOW_ATTRS(win)),
	   GET_WINDOW_PAIR(win)));

	if_EXT_COLORS({
	    if (at & A_COLOR)
		win->_color = PairNumber(at);
	});
	toggle_attr_on(WINDOW_ATTRS(win), at);
	returnCode(OK);
    } else
	returnCode(ERR);
}
