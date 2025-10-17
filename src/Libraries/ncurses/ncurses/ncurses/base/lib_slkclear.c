/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
 *     and: Juergen Pfeifer                         1996-1999               *
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

/*
 *	lib_slkclear.c
 *	Soft key routines.
 *      Remove soft labels from the screen.
 */
#include <curses.priv.h>

MODULE_ID("$Id: lib_slkclear.c,v 1.14 2009/11/07 16:27:05 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_clear) (NCURSES_SP_DCL0)
{
    int rc = ERR;

    T((T_CALLED("slk_clear(%p)"), (void *) SP_PARM));

    if (SP_PARM != 0 && SP_PARM->_slk != 0) {
	SP_PARM->_slk->hidden = TRUE;
	/* For simulated SLK's it looks much more natural to
	   inherit those attributes from the standard screen */
	SP_PARM->_slk->win->_nc_bkgd = StdScreen(SP_PARM)->_nc_bkgd;
	WINDOW_ATTRS(SP_PARM->_slk->win) = WINDOW_ATTRS(StdScreen(SP_PARM));
	if (SP_PARM->_slk->win == StdScreen(SP_PARM)) {
	    rc = OK;
	} else {
	    werase(SP_PARM->_slk->win);
	    rc = wrefresh(SP_PARM->_slk->win);
	}
    }
    returnCode(rc);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_clear(void)
{
    return NCURSES_SP_NAME(slk_clear) (CURRENT_SCREEN);
}
#endif
