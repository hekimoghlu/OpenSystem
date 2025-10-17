/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
 *  Author: Juergen Pfeifer,  1998                                          *
 *     and: Thomas E. Dickey, 2005-on                                       *
 ****************************************************************************/

/*
**	lib_colorset.c
**
**	The routine wcolor_set().
**
*/

#include <curses.priv.h>
#include <ctype.h>

MODULE_ID("$Id: lib_colorset.c,v 1.14 2014/02/01 22:10:42 tom Exp $")

NCURSES_EXPORT(int)
wcolor_set(WINDOW *win, NCURSES_PAIRS_T color_pair_number, void *opts)
{
    int code = ERR;

    T((T_CALLED("wcolor_set(%p,%d)"), (void *) win, (int) color_pair_number));
    if (win
	&& !opts
	&& (SP != 0)
	&& (color_pair_number >= 0)
	&& (color_pair_number < SP->_pair_limit)) {
	TR(TRACE_ATTRS, ("... current %ld", (long) GET_WINDOW_PAIR(win)));
	SET_WINDOW_PAIR(win, color_pair_number);
	if_EXT_COLORS(win->_color = color_pair_number);
	code = OK;
    }
    returnCode(code);
}
