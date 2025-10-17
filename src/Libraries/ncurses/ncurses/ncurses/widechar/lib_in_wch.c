/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
 * Author: Thomas Dickey                                                    *
 ****************************************************************************/

/*
**	lib_in_wch.c
**
**	The routine win_wch().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_in_wch.c,v 1.5 2009/10/24 22:37:55 tom Exp $")

NCURSES_EXPORT(int)
win_wch(WINDOW *win, cchar_t *wcval)
{
    int row, col;
    int code = OK;

    TR(TRACE_CCALLS, (T_CALLED("win_wch(%p,%p)"), (void *) win, (void *) wcval));
    if (win != 0
	&& wcval != 0) {
	getyx(win, row, col);

	*wcval = win->_line[row].text[col];
	TR(TRACE_CCALLS, ("data %s", _tracecchar_t(wcval)));
    } else {
	code = ERR;
    }
    TR(TRACE_CCALLS, (T_RETURN("%d"), code));
    return (code);
}
