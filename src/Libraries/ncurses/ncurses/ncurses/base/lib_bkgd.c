/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
 *     and: Juergen Pfeifer                         1997                    *
 *     and: Sven Verdoolaege                        2000                    *
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: lib_bkgd.c,v 1.49 2014/09/04 09:36:20 tom Exp $")

/*
 * Set the window's background information.
 */
#if USE_WIDEC_SUPPORT
NCURSES_EXPORT(void)
#else
static NCURSES_INLINE void
#endif
wbkgrndset(WINDOW *win, const ARG_CH_T ch)
{
    T((T_CALLED("wbkgdset(%p,%s)"), (void *) win, _tracech_t(ch)));

    if (win) {
	attr_t off = AttrOf(win->_nc_bkgd);
	attr_t on = AttrOf(CHDEREF(ch));

	toggle_attr_off(WINDOW_ATTRS(win), off);
	toggle_attr_on(WINDOW_ATTRS(win), on);

#if NCURSES_EXT_COLORS
	{
	    int pair;

	    if ((pair = GetPair(win->_nc_bkgd)) != 0)
		SET_WINDOW_PAIR(win, 0);
	    if ((pair = GetPair(CHDEREF(ch))) != 0)
		SET_WINDOW_PAIR(win, pair);
	}
#endif

	if (CharOf(CHDEREF(ch)) == L('\0')) {
	    SetChar(win->_nc_bkgd, BLANK_TEXT, AttrOf(CHDEREF(ch)));
	    if_EXT_COLORS(SetPair(win->_nc_bkgd, GetPair(CHDEREF(ch))));
	} else {
	    win->_nc_bkgd = CHDEREF(ch);
	}
#if USE_WIDEC_SUPPORT
	/*
	 * If we're compiled for wide-character support, _bkgrnd is the
	 * preferred location for the background information since it stores
	 * more than _bkgd.  Update _bkgd each time we modify _bkgrnd, so the
	 * macro getbkgd() will work.
	 */
	{
	    cchar_t wch;
	    int tmp;

	    memset(&wch, 0, sizeof(wch));
	    (void) wgetbkgrnd(win, &wch);
	    tmp = _nc_to_char((wint_t) CharOf(wch));

	    win->_bkgd = (((tmp == EOF) ? ' ' : (chtype) tmp)
			  | (AttrOf(wch) & ALL_BUT_COLOR)
			  | (chtype) ColorPair(GET_WINDOW_PAIR(win)));
	}
#endif
    }
    returnVoid;
}

NCURSES_EXPORT(void)
wbkgdset(WINDOW *win, chtype ch)
{
    NCURSES_CH_T wch;
    SetChar2(wch, ch);
    wbkgrndset(win, CHREF(wch));
}

/*
 * Set the window's background information and apply it to each cell.
 */
#if USE_WIDEC_SUPPORT
NCURSES_EXPORT(int)
#else
static NCURSES_INLINE int
#undef wbkgrnd
#endif
wbkgrnd(WINDOW *win, const ARG_CH_T ch)
{
    int code = ERR;
    int x, y;

    T((T_CALLED("wbkgd(%p,%s)"), (void *) win, _tracech_t(ch)));

    if (win) {
	NCURSES_CH_T new_bkgd = CHDEREF(ch);
	NCURSES_CH_T old_bkgrnd;

	memset(&old_bkgrnd, 0, sizeof(old_bkgrnd));
	(void) wgetbkgrnd(win, &old_bkgrnd);

	(void) wbkgrndset(win, CHREF(new_bkgd));
	(void) wattrset(win, (int) AttrOf(win->_nc_bkgd));

	for (y = 0; y <= win->_maxy; y++) {
	    for (x = 0; x <= win->_maxx; x++) {
		if (CharEq(win->_line[y].text[x], old_bkgrnd)) {
		    win->_line[y].text[x] = win->_nc_bkgd;
		} else {
		    NCURSES_CH_T wch = win->_line[y].text[x];
		    RemAttr(wch, (~(A_ALTCHARSET | A_CHARTEXT)));
		    win->_line[y].text[x] = _nc_render(win, wch);
		}
	    }
	}
	touchwin(win);
	_nc_synchook(win);
	code = OK;
    }
    returnCode(code);
}

NCURSES_EXPORT(int)
wbkgd(WINDOW *win, chtype ch)
{
    NCURSES_CH_T wch;
    SetChar2(wch, ch);
    return wbkgrnd(win, CHREF(wch));
}
