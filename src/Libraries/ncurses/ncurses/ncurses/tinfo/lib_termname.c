/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#include <curses.priv.h>

MODULE_ID("$Id: lib_termname.c,v 1.12 2009/10/24 21:56:58 tom Exp $")

NCURSES_EXPORT(char *)
NCURSES_SP_NAME(termname) (NCURSES_SP_DCL0)
{
    char *name = 0;

    T((T_CALLED("termname(%p)"), (void *) SP_PARM));

#if NCURSES_SP_FUNCS
    if (TerminalOf(SP_PARM) != 0) {
	name = TerminalOf(SP_PARM)->_termname;
    }
#else
    if (cur_term != 0)
	name = cur_term->_termname;
#endif

    returnPtr(name);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char *)
termname(void)
{
    return NCURSES_SP_NAME(termname) (CURRENT_SCREEN);
}
#endif
