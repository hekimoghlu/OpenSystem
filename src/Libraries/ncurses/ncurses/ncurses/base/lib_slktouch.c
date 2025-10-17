/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
 *  Author: Juergen Pfeifer                         1997,2009               *
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

/*
 *	lib_slktouch.c
 *	Soft key routines.
 *      Force the code to believe that the soft keys have been changed.
 */
#include <curses.priv.h>

MODULE_ID("$Id: lib_slktouch.c,v 1.8 2009/10/24 22:12:21 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_touch) (NCURSES_SP_DCL0)
{
    T((T_CALLED("slk_touch(%p)"), (void *) SP_PARM));

    if (SP_PARM == 0 || SP_PARM->_slk == 0)
	returnCode(ERR);
    SP_PARM->_slk->dirty = TRUE;

    returnCode(OK);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_touch(void)
{
    return NCURSES_SP_NAME(slk_touch) (CURRENT_SCREEN);
}
#endif
