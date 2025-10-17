/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
**	lib_longname.c
**
**	The routine longname().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_longname.c,v 1.13 2015/07/25 20:08:14 tom Exp $")

#if USE_REENTRANT
NCURSES_EXPORT(char *)
NCURSES_SP_NAME(longname) (NCURSES_SP_DCL0)
{
    static char empty[] =
    {'\0'};
    char *ptr;

    T((T_CALLED("longname(%p)"), (void *) SP_PARM));

    if (SP_PARM) {
	for (ptr = SP_PARM->_ttytype + strlen(SP_PARM->_ttytype);
	     ptr > SP_PARM->_ttytype;
	     ptr--)
	    if (*ptr == '|')
		returnPtr(ptr + 1);
	returnPtr(SP_PARM->_ttytype);
    }
    return empty;
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char *)
longname(void)
{
    return NCURSES_SP_NAME(longname) (CURRENT_SCREEN);
}
#endif

#else

/* a dummy entrypoint is simpler than generating a conditional in curses.h */
#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char *)
NCURSES_SP_NAME(longname) (NCURSES_SP_DCL0)
{
    (void) SP_PARM;
    return longname();
}
#endif

NCURSES_EXPORT(char *)
longname(void)
{
    char *ptr;

    T((T_CALLED("longname()")));

    for (ptr = ttytype + strlen(ttytype);
	 ptr > ttytype;
	 ptr--)
	if (*ptr == '|')
	    returnPtr(ptr + 1);
    returnPtr(ttytype);
}
#endif
