/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
 *  Author: Thomas E. Dickey          1998-on                               *
 *          Juergen Pfeifer           2009                                  *
 ****************************************************************************/

#include <curses.priv.h>

#ifndef CUR
#define CUR SP_TERMTYPE
#endif

MODULE_ID("$Id: lib_dft_fgbg.c,v 1.27 2014/03/08 20:32:59 tom Exp $")

/*
 * Modify the behavior of color-pair 0 so that the library doesn't assume that
 * it is white on black.  This is an extension to XSI curses.
 */
NCURSES_EXPORT(int)
NCURSES_SP_NAME(use_default_colors) (NCURSES_SP_DCL0)
{
    T((T_CALLED("use_default_colors(%p)"), (void *) SP_PARM));
    returnCode(NCURSES_SP_NAME(assume_default_colors) (NCURSES_SP_ARGx -1, -1));
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
use_default_colors(void)
{
    return NCURSES_SP_NAME(use_default_colors) (CURRENT_SCREEN);
}
#endif

/*
 * Modify the behavior of color-pair 0 so that the library assumes that it
 * is something specific, possibly not white on black.
 */
NCURSES_EXPORT(int)
NCURSES_SP_NAME(assume_default_colors) (NCURSES_SP_DCLx int fg, int bg)
{
    int code = ERR;

    T((T_CALLED("assume_default_colors(%p,%d,%d)"), (void *) SP_PARM, fg, bg));
#ifdef USE_TERM_DRIVER
    if (sp != 0)
	code = CallDriver_2(sp, td_defaultcolors, fg, bg);
#else
    if ((orig_pair || orig_colors) && !initialize_pair) {

	SP_PARM->_default_color = isDefaultColor(fg) || isDefaultColor(bg);
	SP_PARM->_has_sgr_39_49 = (tigetflag("AX") == TRUE);
	SP_PARM->_default_fg = isDefaultColor(fg) ? COLOR_DEFAULT : (fg & C_MASK);
	SP_PARM->_default_bg = isDefaultColor(bg) ? COLOR_DEFAULT : (bg & C_MASK);
	if (SP_PARM->_color_pairs != 0) {
	    bool save = SP_PARM->_default_color;
	    SP_PARM->_assumed_color = TRUE;
	    SP_PARM->_default_color = TRUE;
	    init_pair(0, (short) fg, (short) bg);
	    SP_PARM->_default_color = save;
	}
	code = OK;
    }
#endif
    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
assume_default_colors(int fg, int bg)
{
    return NCURSES_SP_NAME(assume_default_colors) (CURRENT_SCREEN, fg, bg);
}
#endif
