/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
 *  Author:  Juergen Pfeifer, 1998                                          *
 *     and:  Thomas E. Dickey 2005-on                                       *
 ****************************************************************************/

/*
 *	lib_slkatr_set.c
 *	Soft key routines.
 *	Set the label's attributes
 */
#include <curses.priv.h>

MODULE_ID("$Id: lib_slkatr_set.c,v 1.15 2014/02/01 22:10:42 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_attr_set) (NCURSES_SP_DCLx
			       const attr_t attr,
			       NCURSES_PAIRS_T color_pair_number,
			       void *opts)
{
    int code = ERR;

    T((T_CALLED("slk_attr_set(%p,%s,%d)"),
       (void *) SP_PARM,
       _traceattr(attr),
       (int) color_pair_number));

    if (SP_PARM != 0
	&& SP_PARM->_slk != 0
	&& !opts
	&& color_pair_number >= 0
	&& color_pair_number < SP_PARM->_pair_limit) {
	TR(TRACE_ATTRS, ("... current %s", _tracech_t(CHREF(SP_PARM->_slk->attr))));
	SetAttr(SP_PARM->_slk->attr, attr);
	if (color_pair_number > 0) {
	    SetPair(SP_PARM->_slk->attr, color_pair_number);
	}
	TR(TRACE_ATTRS, ("new attribute is %s", _tracech_t(CHREF(SP_PARM->_slk->attr))));
	code = OK;
    }
    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_attr_set(const attr_t attr, NCURSES_COLOR_T color_pair_number, void *opts)
{
    return NCURSES_SP_NAME(slk_attr_set) (CURRENT_SCREEN, attr,
					  color_pair_number, opts);
}
#endif
