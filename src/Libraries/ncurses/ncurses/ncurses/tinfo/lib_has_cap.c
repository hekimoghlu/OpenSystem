/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
 *     and: Thomas E. Dickey                        1996-2003               *
 *     and: Juergen Pfeifer                         2009                    *
 ****************************************************************************/

/*
**	lib_has_cap.c
**
**	The routines to query terminal capabilities
**
*/

#include <curses.priv.h>

#ifndef CUR
#define CUR SP_TERMTYPE
#endif

MODULE_ID("$Id: lib_has_cap.c,v 1.10 2013/11/16 19:57:22 tom Exp $")

NCURSES_EXPORT(bool)
NCURSES_SP_NAME(has_ic) (NCURSES_SP_DCL0)
{
    bool code = FALSE;

    T((T_CALLED("has_ic(%p)"), (void *) SP_PARM));

    if (HasTInfoTerminal(SP_PARM)) {
	code = ((insert_character || parm_ich
		 || (enter_insert_mode && exit_insert_mode))
		&& (delete_character || parm_dch)) ? TRUE : FALSE;
    }

    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(bool)
has_ic(void)
{
    return NCURSES_SP_NAME(has_ic) (CURRENT_SCREEN);
}
#endif

NCURSES_EXPORT(bool)
NCURSES_SP_NAME(has_il) (NCURSES_SP_DCL0)
{
    bool code = FALSE;
    T((T_CALLED("has_il(%p)"), (void *) SP_PARM));
    if (HasTInfoTerminal(SP_PARM)) {
	code = ((insert_line || parm_insert_line)
		&& (delete_line || parm_delete_line)) ? TRUE : FALSE;
    }

    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(bool)
has_il(void)
{
    return NCURSES_SP_NAME(has_il) (CURRENT_SCREEN);
}
#endif
