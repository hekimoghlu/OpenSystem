/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
 *  Author:  Juergen Pfeifer, 1997                                          *
 *     and:  Thomas E. Dickey 2005                                          *
 ****************************************************************************/

/*
 *	lib_slkatron.c
 *	Soft key routines.
 *      Switch on labels attributes
 */
#include <curses.priv.h>

MODULE_ID("$Id: lib_slkatron.c,v 1.12 2010/03/31 23:38:02 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_attron) (NCURSES_SP_DCLx const chtype attr)
{
    T((T_CALLED("slk_attron(%p,%s)"), (void *) SP_PARM, _traceattr(attr)));

    if (SP_PARM != 0 && SP_PARM->_slk != 0) {
	TR(TRACE_ATTRS, ("... current %s", _tracech_t(CHREF(SP_PARM->_slk->attr))));
	AddAttr(SP_PARM->_slk->attr, attr);
	if ((attr & A_COLOR) != 0) {
	    SetPair(SP_PARM->_slk->attr, PairNumber(attr));
	}
	TR(TRACE_ATTRS, ("new attribute is %s", _tracech_t(CHREF(SP_PARM->_slk->attr))));
	returnCode(OK);
    } else
	returnCode(ERR);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_attron(const chtype attr)
{
    return NCURSES_SP_NAME(slk_attron) (CURRENT_SCREEN, attr);
}
#endif
