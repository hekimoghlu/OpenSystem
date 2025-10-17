/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
 * Authors: Sven Verdoolaege and Thomas Dickey 2001,2002                    *
 ****************************************************************************/

/*
**	lib_box_set.c
**
**	The routine wborder_set().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_box_set.c,v 1.6 2011/06/25 19:02:07 Vassili.Courzakis Exp $")

NCURSES_EXPORT(int)
wborder_set(WINDOW *win,
	    const ARG_CH_T ls, const ARG_CH_T rs,
	    const ARG_CH_T ts, const ARG_CH_T bs,
	    const ARG_CH_T tl, const ARG_CH_T tr,
	    const ARG_CH_T bl, const ARG_CH_T br)
{
    NCURSES_SIZE_T i;
    NCURSES_SIZE_T endx, endy;
    NCURSES_CH_T wls, wrs, wts, wbs, wtl, wtr, wbl, wbr;

    T((T_CALLED("wborder_set(%p,%s,%s,%s,%s,%s,%s,%s,%s)"),
       (void *) win,
       _tracech_t2(1, ls),
       _tracech_t2(2, rs),
       _tracech_t2(3, ts),
       _tracech_t2(4, bs),
       _tracech_t2(5, tl),
       _tracech_t2(6, tr),
       _tracech_t2(7, bl),
       _tracech_t2(8, br)));

    if (!win)
	returnCode(ERR);

#define RENDER_WITH_DEFAULT(ch,def) w ##ch = _nc_render(win, (ch == 0) ? *(const ARG_CH_T)def : *ch)

    RENDER_WITH_DEFAULT(ls, WACS_VLINE);
    RENDER_WITH_DEFAULT(rs, WACS_VLINE);
    RENDER_WITH_DEFAULT(ts, WACS_HLINE);
    RENDER_WITH_DEFAULT(bs, WACS_HLINE);
    RENDER_WITH_DEFAULT(tl, WACS_ULCORNER);
    RENDER_WITH_DEFAULT(tr, WACS_URCORNER);
    RENDER_WITH_DEFAULT(bl, WACS_LLCORNER);
    RENDER_WITH_DEFAULT(br, WACS_LRCORNER);

    T(("using %s, %s, %s, %s, %s, %s, %s, %s",
       _tracech_t2(1, CHREF(wls)),
       _tracech_t2(2, CHREF(wrs)),
       _tracech_t2(3, CHREF(wts)),
       _tracech_t2(4, CHREF(wbs)),
       _tracech_t2(5, CHREF(wtl)),
       _tracech_t2(6, CHREF(wtr)),
       _tracech_t2(7, CHREF(wbl)),
       _tracech_t2(8, CHREF(wbr))));

    endx = win->_maxx;
    endy = win->_maxy;

    for (i = 0; i <= endx; i++) {
	win->_line[0].text[i] = wts;
	win->_line[endy].text[i] = wbs;
    }
    win->_line[endy].firstchar = win->_line[0].firstchar = 0;
    win->_line[endy].lastchar = win->_line[0].lastchar = endx;

    for (i = 0; i <= endy; i++) {
	win->_line[i].text[0] = wls;
	win->_line[i].text[endx] = wrs;
	win->_line[i].firstchar = 0;
	win->_line[i].lastchar = endx;
    }
    win->_line[0].text[0] = wtl;
    win->_line[0].text[endx] = wtr;
    win->_line[endy].text[0] = wbl;
    win->_line[endy].text[endx] = wbr;

    _nc_synchook(win);
    returnCode(OK);
}
