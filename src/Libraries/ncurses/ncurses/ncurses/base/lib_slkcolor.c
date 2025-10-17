/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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
 *  Author:  Juergen Pfeifer, 1998,2009                                     *
 *     and:  Thomas E. Dickey 2005-on                                       *
 ****************************************************************************/

/*
 *	lib_slkcolor.c
 *	Soft key routines.
 *	Set the label's color
 */
#include <curses.priv.h>

MODULE_ID("$Id: lib_slkcolor.c,v 1.17 2014/02/01 22:10:42 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_color) (NCURSES_SP_DCLx NCURSES_PAIRS_T color_pair_number)
{
    int code = ERR;

    T((T_CALLED("slk_color(%p,%d)"), (void *) SP_PARM, (int) color_pair_number));

    if (SP_PARM != 0
	&& SP_PARM->_slk != 0
	&& color_pair_number >= 0
	&& color_pair_number < SP_PARM->_pair_limit) {
	TR(TRACE_ATTRS, ("... current is %s", _tracech_t(CHREF(SP_PARM->_slk->attr))));
	SetPair(SP_PARM->_slk->attr, color_pair_number);
	TR(TRACE_ATTRS, ("new attribute is %s", _tracech_t(CHREF(SP_PARM->_slk->attr))));
	code = OK;
    }
    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_color(NCURSES_PAIRS_T color_pair_number)
{
    return NCURSES_SP_NAME(slk_color) (CURRENT_SCREEN, color_pair_number);
}
#endif
