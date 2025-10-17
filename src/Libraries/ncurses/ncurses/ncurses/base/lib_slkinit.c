/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
 *     and: Juergen Pfeifer                         2009                    *
 ****************************************************************************/

/*
 *	lib_slkinit.c
 *	Soft key routines.
 *      Initialize soft labels.  Called by the user before initscr().
 */
#include <curses.priv.h>

MODULE_ID("$Id: lib_slkinit.c,v 1.13 2009/10/31 00:10:46 tom Exp $")

#ifdef USE_SP_RIPOFF
#define SoftkeyFormat SP_PARM->slk_format
#else
#define SoftkeyFormat _nc_globals.slk_format
#endif

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_init) (NCURSES_SP_DCLx int format)
{
    int code = ERR;

    START_TRACE();
    T((T_CALLED("slk_init(%p,%d)"), (void *) SP_PARM, format));

    if (format >= 0
	&& format <= 3
#ifdef USE_SP_RIPOFF
	&& SP_PARM
	&& SP_PARM->_prescreen
#endif
	&& !SoftkeyFormat) {
	SoftkeyFormat = 1 + format;
	code = NCURSES_SP_NAME(_nc_ripoffline) (NCURSES_SP_ARGx
						-SLK_LINES(SoftkeyFormat),
						_nc_slk_initialize);
    }
    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_init(int format)
{
    return NCURSES_SP_NAME(slk_init) (CURRENT_SCREEN_PRE, format);
}
#endif
