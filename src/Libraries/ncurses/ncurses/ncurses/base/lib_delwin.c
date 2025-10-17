/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
 *     and: Juergen Pfeifer                         2008                    *
 ****************************************************************************/

/*
**	lib_delwin.c
**
**	The routine delwin().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_delwin.c,v 1.20 2009/10/24 22:02:14 tom Exp $")

static bool
cannot_delete(WINDOW *win)
{
    WINDOWLIST *p;
    bool result = TRUE;
#ifdef USE_SP_WINDOWLIST
    SCREEN *sp = _nc_screen_of(win);
#endif

    for (each_window(SP_PARM, p)) {
	if (&(p->win) == win) {
	    result = FALSE;
	} else if ((p->win._flags & _SUBWIN) != 0
		   && p->win._parent == win) {
	    result = TRUE;
	    break;
	}
    }
    return result;
}

NCURSES_EXPORT(int)
delwin(WINDOW *win)
{
    int result = ERR;

    T((T_CALLED("delwin(%p)"), (void *) win));

    if (_nc_try_global(curses) == 0) {
	if (win == 0
	    || cannot_delete(win)) {
	    result = ERR;
	} else {
#if NCURSES_SP_FUNCS
	    SCREEN *sp = _nc_screen_of(win);
#endif
	    if (win->_flags & _SUBWIN)
		touchwin(win->_parent);
	    else if (CurScreen(SP_PARM) != 0)
		touchwin(CurScreen(SP_PARM));

	    result = _nc_freewin(win);
	}
	_nc_unlock_global(curses);
    }
    returnCode(result);
}
