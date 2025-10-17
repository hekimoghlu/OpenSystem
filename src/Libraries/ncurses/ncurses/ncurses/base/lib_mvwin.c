/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
 *     and: Juergen Pfeifer                                                 *
 ****************************************************************************/

/*
**	lib_mvwin.c
**
**	The routine mvwin().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_mvwin.c,v 1.18 2010/12/19 01:22:58 tom Exp $")

NCURSES_EXPORT(int)
mvwin(WINDOW *win, int by, int bx)
{
#if NCURSES_SP_FUNCS
    SCREEN *sp = _nc_screen_of(win);
#endif

    T((T_CALLED("mvwin(%p,%d,%d)"), (void *) win, by, bx));

    if (!win || (win->_flags & _ISPAD))
	returnCode(ERR);

    /*
     * mvwin() should only modify the indices.  See test/demo_menus.c and
     * test/movewindow.c for examples.
     */
#if 0
    /* Copying subwindows is allowed, but it is expensive... */
    if (win->_flags & _SUBWIN) {
	int err = ERR;
	WINDOW *parent = win->_parent;
	if (parent) {		/* Now comes the complicated and costly part, you should really
				 * try to avoid to move subwindows. Because a subwindow shares
				 * the text buffers with its parent, one can't do a simple
				 * memmove of the text buffers. One has to create a copy, then
				 * to relocate the subwindow and then to do a copy.
				 */
	    if ((by - parent->_begy == win->_pary) &&
		(bx - parent->_begx == win->_parx))
		err = OK;	/* we don't actually move */
	    else {
		WINDOW *clone = dupwin(win);
		if (clone) {
		    /* now we have the clone, so relocate win */

		    werase(win);	/* Erase the original place     */
		    /* fill with parents background */
		    wbkgrnd(win, CHREF(parent->_nc_bkgd));
		    wsyncup(win);	/* Tell the parent(s)           */

		    err = mvderwin(win,
				   by - parent->_begy,
				   bx - parent->_begx);
		    if (err != ERR) {
			err = copywin(clone, win,
				      0, 0, 0, 0, win->_maxy, win->_maxx, 0);
			if (ERR != err)
			    wsyncup(win);
		    }
		    if (ERR == delwin(clone))
			err = ERR;
		}
	    }
	}
	returnCode(err);
    }
#endif

    if (by + win->_maxy > screen_lines(SP_PARM) - 1
	|| bx + win->_maxx > screen_columns(SP_PARM) - 1
	|| by < 0
	|| bx < 0)
	returnCode(ERR);

    /*
     * Whether or not the window is moved, touch the window's contents so
     * that a following call to 'wrefresh()' will paint the window at the
     * new location.  This ensures that if the caller has refreshed another
     * window at the same location, that this one will be displayed.
     */
    win->_begy = (NCURSES_SIZE_T) by;
    win->_begx = (NCURSES_SIZE_T) bx;
    returnCode(touchwin(win));
}
