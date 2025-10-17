/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
 *	flash.c
 *
 *	The routine flash().
 *
 */

#include <curses.priv.h>

#ifndef CUR
#define CUR SP_TERMTYPE
#endif

MODULE_ID("$Id: lib_flash.c,v 1.14 2014/03/08 20:32:59 tom Exp $")

/*
 *	flash()
 *
 *	Flash the current terminal's screen if possible.   If not,
 *	sound the audible bell if one exists.
 *
 */

NCURSES_EXPORT(int)
NCURSES_SP_NAME(flash) (NCURSES_SP_DCL0)
{
    int res = ERR;

    T((T_CALLED("flash(%p)"), (void *) SP_PARM));
#ifdef USE_TERM_DRIVER
    if (SP_PARM != 0)
	res = CallDriver_1(SP_PARM, td_doBeepOrFlash, FALSE);
#else
    if (HasTerminal(SP_PARM)) {
	/* FIXME: should make sure that we are not in altchar mode */
	if (flash_screen) {
	    res = NCURSES_PUTP2_FLUSH("flash_screen", flash_screen);
	} else if (bell) {
	    res = NCURSES_PUTP2_FLUSH("bell", bell);
	}
    }
#endif
    returnCode(res);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
flash(void)
{
    return NCURSES_SP_NAME(flash) (CURRENT_SCREEN);
}
#endif
