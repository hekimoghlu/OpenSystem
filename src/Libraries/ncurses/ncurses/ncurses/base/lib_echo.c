/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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
 *	echo.c
 *
 *	Routines:
 *		echo()
 *		noecho()
 *
 */

#include <curses.priv.h>

MODULE_ID("$Id: lib_echo.c,v 1.8 2009/10/24 22:02:14 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(echo) (NCURSES_SP_DCL0)
{
    T((T_CALLED("echo(%p)"), (void *) SP_PARM));
    if (0 == SP_PARM)
	returnCode(ERR);
    SP_PARM->_echo = TRUE;
    returnCode(OK);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
echo(void)
{
    return NCURSES_SP_NAME(echo) (CURRENT_SCREEN);
}
#endif

NCURSES_EXPORT(int)
NCURSES_SP_NAME(noecho) (NCURSES_SP_DCL0)
{
    T((T_CALLED("noecho(%p)"), (void *) SP_PARM));
    if (0 == SP_PARM)
	returnCode(ERR);
    SP_PARM->_echo = FALSE;
    returnCode(OK);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
noecho(void)
{
    return NCURSES_SP_NAME(noecho) (CURRENT_SCREEN);
}
#endif
