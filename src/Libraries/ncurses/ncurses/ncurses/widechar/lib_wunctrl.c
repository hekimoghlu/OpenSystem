/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
/*
**	lib_wunctrl.c
**
**	The routine wunctrl().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_wunctrl.c,v 1.16 2012/12/15 20:53:42 tom Exp $")

NCURSES_EXPORT(wchar_t *)
NCURSES_SP_NAME(wunctrl) (NCURSES_SP_DCLx cchar_t *wc)
{
    static wchar_t str[CCHARW_MAX + 1], *wsp;
    wchar_t *result;

    if (wc == 0) {
	result = 0;
    } else if (SP_PARM != 0 && Charable(*wc)) {
	const char *p =
	NCURSES_SP_NAME(unctrl) (NCURSES_SP_ARGx
				 (unsigned) _nc_to_char((wint_t)CharOf(*wc)));

	for (wsp = str; *p; ++p) {
	    *wsp++ = (wchar_t) _nc_to_widechar(*p);
	}
	*wsp = 0;
	result = str;
    } else {
	result = wc->chars;
    }
    return result;
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(wchar_t *)
wunctrl(cchar_t *wc)
{
    return NCURSES_SP_NAME(wunctrl) (CURRENT_SCREEN, wc);
}
#endif
