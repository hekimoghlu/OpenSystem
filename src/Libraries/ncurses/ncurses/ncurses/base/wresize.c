/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
 *  Author: Thomas E. Dickey 1996-on                                        *
 *     and: Juergen Pfeifer                                                 *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: wresize.c,v 1.35 2011/05/21 18:55:07 tom Exp $")

static int
cleanup_lines(struct ldat *data, int length)
{
    while (--length >= 0)
	free(data[length].text);
    free(data);
    return ERR;
}

/*
 * If we have reallocated the ldat structs, we will have to repair pointers
 * used in subwindows.
 */
static void
repair_subwindows(WINDOW *cmp)
{
    WINDOWLIST *wp;
    struct ldat *pline = cmp->_line;
    int row;
#ifdef USE_SP_WINDOWLIST
    SCREEN *sp = _nc_screen_of(cmp);
#endif

    _nc_lock_global(curses);

    for (each_window(SP_PARM, wp)) {
	WINDOW *tst = &(wp->win);

	if (tst->_parent == cmp) {

	    if (tst->_pary > cmp->_maxy)
		tst->_pary = cmp->_maxy;
	    if (tst->_parx > cmp->_maxx)
		tst->_parx = cmp->_maxx;

	    if (tst->_maxy + tst->_pary > cmp->_maxy)
		tst->_maxy = (NCURSES_SIZE_T) (cmp->_maxy - tst->_pary);
	    if (tst->_maxx + tst->_parx > cmp->_maxx)
		tst->_maxx = (NCURSES_SIZE_T) (cmp->_maxx - tst->_parx);

	    for (row = 0; row <= tst->_maxy; ++row) {
		tst->_line[row].text = &pline[tst->_pary + row].text[tst->_parx];
	    }
	    repair_subwindows(tst);
	}
    }
    _nc_unlock_global(curses);
}

/*
 * Reallocate a curses WINDOW struct to either shrink or grow to the specified
 * new lines/columns.  If it grows, the new character cells are filled with
 * blanks.  The application is responsible for repainting the blank area.
 */
NCURSES_EXPORT(int)
wresize(WINDOW *win, int ToLines, int ToCols)
{
    int col, row, size_x, size_y;
    struct ldat *pline;
    struct ldat *new_lines = 0;

#ifdef TRACE
    T((T_CALLED("wresize(%p,%d,%d)"), (void *) win, ToLines, ToCols));
    if (win) {
	TR(TRACE_UPDATE, ("...beg (%ld, %ld), max(%ld,%ld), reg(%ld,%ld)",
			  (long) win->_begy, (long) win->_begx,
			  (long) win->_maxy, (long) win->_maxx,
			  (long) win->_regtop, (long) win->_regbottom));
	if (USE_TRACEF(TRACE_UPDATE)) {
	    _tracedump("...before", win);
	    _nc_unlock_global(tracef);
	}
    }
#endif

    if (!win || --ToLines < 0 || --ToCols < 0)
	returnCode(ERR);

    size_x = win->_maxx;
    size_y = win->_maxy;

    if (ToLines == size_y
	&& ToCols == size_x)
	returnCode(OK);

    if ((win->_flags & _SUBWIN)) {
	/*
	 * Check if the new limits will fit into the parent window's size.  If
	 * not, do not resize.  We could adjust the location of the subwindow,
	 * but the application may not like that.
	 */
	if (win->_pary + ToLines > win->_parent->_maxy
	    || win->_parx + ToCols > win->_parent->_maxx) {
	    returnCode(ERR);
	}
	pline = win->_parent->_line;
    } else {
	pline = 0;
    }

    /*
     * Allocate new memory as needed.  Do the allocations without modifying
     * the original window, in case an allocation fails.  Always allocate
     * (at least temporarily) the array pointing to the individual lines.
     */
    new_lines = typeCalloc(struct ldat, (unsigned) (ToLines + 1));
    if (new_lines == 0)
	returnCode(ERR);

    /*
     * For each line in the target, allocate or adjust pointers for the
     * corresponding text, depending on whether this is a window or a
     * subwindow.
     */
    for (row = 0; row <= ToLines; ++row) {
	int begin = (row > size_y) ? 0 : (size_x + 1);
	int end = ToCols;
	NCURSES_CH_T *s;

	if (!(win->_flags & _SUBWIN)) {
	    if (row <= size_y) {
		if (ToCols != size_x) {
		    s = typeMalloc(NCURSES_CH_T, (unsigned) ToCols + 1);
		    if (s == 0)
			returnCode(cleanup_lines(new_lines, row));
		    for (col = 0; col <= ToCols; ++col) {
			s[col] = (col <= size_x
				  ? win->_line[row].text[col]
				  : win->_nc_bkgd);
		    }
		} else {
		    s = win->_line[row].text;
		}
	    } else {
		s = typeMalloc(NCURSES_CH_T, (unsigned) ToCols + 1);
		if (s == 0)
		    returnCode(cleanup_lines(new_lines, row));
		for (col = 0; col <= ToCols; ++col)
		    s[col] = win->_nc_bkgd;
	    }
	} else if (pline != 0 && pline[win->_pary + row].text != 0) {
	    s = &pline[win->_pary + row].text[win->_parx];
	} else {
	    s = 0;
	}

	if_USE_SCROLL_HINTS(new_lines[row].oldindex = row);
	if (row <= size_y) {
	    new_lines[row].firstchar = win->_line[row].firstchar;
	    new_lines[row].lastchar = win->_line[row].lastchar;
	}
	if ((ToCols != size_x) || (row > size_y)) {
	    if (end >= begin) {	/* growing */
		if (new_lines[row].firstchar < begin)
		    new_lines[row].firstchar = (NCURSES_SIZE_T) begin;
	    } else {		/* shrinking */
		new_lines[row].firstchar = 0;
	    }
	    new_lines[row].lastchar = (NCURSES_SIZE_T) ToCols;
	}
	new_lines[row].text = s;
    }

    /*
     * Dispose of unwanted memory.
     */
    if (!(win->_flags & _SUBWIN)) {
	if (ToCols == size_x) {
	    for (row = ToLines + 1; row <= size_y; row++) {
		free(win->_line[row].text);
	    }
	} else {
	    for (row = 0; row <= size_y; row++) {
		free(win->_line[row].text);
	    }
	}
    }

    free(win->_line);
    win->_line = new_lines;

    /*
     * Finally, adjust the parameters showing screen size and cursor
     * position:
     */
    win->_maxx = (NCURSES_SIZE_T) ToCols;
    win->_maxy = (NCURSES_SIZE_T) ToLines;

    if (win->_regtop > win->_maxy)
	win->_regtop = win->_maxy;
    if (win->_regbottom > win->_maxy
	|| win->_regbottom == size_y)
	win->_regbottom = win->_maxy;

    if (win->_curx > win->_maxx)
	win->_curx = win->_maxx;
    if (win->_cury > win->_maxy)
	win->_cury = win->_maxy;

    /*
     * Check for subwindows of this one, and readjust pointers to our text,
     * if needed.
     */
    repair_subwindows(win);

#ifdef TRACE
    TR(TRACE_UPDATE, ("...beg (%ld, %ld), max(%ld,%ld), reg(%ld,%ld)",
		      (long) win->_begy, (long) win->_begx,
		      (long) win->_maxy, (long) win->_maxx,
		      (long) win->_regtop, (long) win->_regbottom));
    if (USE_TRACEF(TRACE_UPDATE)) {
	_tracedump("...after:", win);
	_nc_unlock_global(tracef);
    }
#endif
    returnCode(OK);
}
