/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
**	lib_endwin.c
**
**	The routine endwin().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_endwin.c,v 1.23 2014/03/08 20:32:59 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(endwin) (NCURSES_SP_DCL0)
{
    int code = ERR;

    T((T_CALLED("endwin(%p)"), (void *) SP_PARM));

    if (SP_PARM) {
#ifdef USE_TERM_DRIVER
	TERMINAL_CONTROL_BLOCK *TCB = TCBOf(SP_PARM);

	SP_PARM->_endwin = TRUE;
	if (TCB && TCB->drv && TCB->drv->td_scexit)
	    TCB->drv->td_scexit(SP_PARM);
#else
	SP_PARM->_endwin = TRUE;
	SP_PARM->_mouse_wrap(SP_PARM);
	_nc_screen_wrap();
	_nc_mvcur_wrap();	/* wrap up cursor addressing */
#endif
	code = NCURSES_SP_NAME(reset_shell_mode) (NCURSES_SP_ARG);
    }

    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
endwin(void)
{
    return NCURSES_SP_NAME(endwin) (CURRENT_SCREEN);
}
#endif
