/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
 *     and: Sven Verdoolaege                        2001                    *
 ****************************************************************************/

/*
**	lib_box.c
**
**	The routine wborder().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_box.c,v 1.24 2010/04/24 23:51:57 tom Exp $")

#if USE_WIDEC_SUPPORT
static NCURSES_INLINE chtype
_my_render(WINDOW *win, chtype ch)
{
    NCURSES_CH_T wch;
    SetChar2(wch, ch);
    wch = _nc_render(win, wch);
    return ((attr_t) CharOf(wch)) | AttrOf(wch);
}

#define RENDER_WITH_DEFAULT(ch,def) w ## ch = _my_render(win, (ch == 0) ? def : ch)
#else
#define RENDER_WITH_DEFAULT(ch,def) w ## ch = _nc_render(win, (ch == 0) ? def : ch)
#endif

NCURSES_EXPORT(int)
wborder(WINDOW *win,
	chtype ls, chtype rs,
	chtype ts, chtype bs,
	chtype tl, chtype tr,
	chtype bl, chtype br)
{
    NCURSES_SIZE_T i;
    NCURSES_SIZE_T endx, endy;
    chtype wls, wrs, wts, wbs, wtl, wtr, wbl, wbr;

    T((T_CALLED("wborder(%p,%s,%s,%s,%s,%s,%s,%s,%s)"),
       (void *) win,
       _tracechtype2(1, ls),
       _tracechtype2(2, rs),
       _tracechtype2(3, ts),
       _tracechtype2(4, bs),
       _tracechtype2(5, tl),
       _tracechtype2(6, tr),
       _tracechtype2(7, bl),
       _tracechtype2(8, br)));

    if (!win)
	returnCode(ERR);

    RENDER_WITH_DEFAULT(ls, ACS_VLINE);
    RENDER_WITH_DEFAULT(rs, ACS_VLINE);
    RENDER_WITH_DEFAULT(ts, ACS_HLINE);
    RENDER_WITH_DEFAULT(bs, ACS_HLINE);
    RENDER_WITH_DEFAULT(tl, ACS_ULCORNER);
    RENDER_WITH_DEFAULT(tr, ACS_URCORNER);
    RENDER_WITH_DEFAULT(bl, ACS_LLCORNER);
    RENDER_WITH_DEFAULT(br, ACS_LRCORNER);

    T(("using %s, %s, %s, %s, %s, %s, %s, %s",
       _tracechtype2(1, wls),
       _tracechtype2(2, wrs),
       _tracechtype2(3, wts),
       _tracechtype2(4, wbs),
       _tracechtype2(5, wtl),
       _tracechtype2(6, wtr),
       _tracechtype2(7, wbl),
       _tracechtype2(8, wbr)));

    endx = win->_maxx;
    endy = win->_maxy;

    for (i = 0; i <= endx; i++) {
	SetChar2(win->_line[0].text[i], wts);
	SetChar2(win->_line[endy].text[i], wbs);
    }
    win->_line[endy].firstchar = win->_line[0].firstchar = 0;
    win->_line[endy].lastchar = win->_line[0].lastchar = endx;

    for (i = 0; i <= endy; i++) {
	SetChar2(win->_line[i].text[0], wls);
	SetChar2(win->_line[i].text[endx], wrs);
	win->_line[i].firstchar = 0;
	win->_line[i].lastchar = endx;
    }
    SetChar2(win->_line[0].text[0], wtl);
    SetChar2(win->_line[0].text[endx], wtr);
    SetChar2(win->_line[endy].text[0], wbl);
    SetChar2(win->_line[endy].text[endx], wbr);

    _nc_synchook(win);
    returnCode(OK);
}
