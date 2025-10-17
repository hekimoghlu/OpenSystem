/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
 *	lib_slk_wset.c
 *      Set soft label text.
 */
#include <curses.priv.h>

#if HAVE_WCTYPE_H
#include <wctype.h>
#endif

MODULE_ID("$Id: lib_slk_wset.c,v 1.13 2011/10/22 15:52:20 tom Exp $")

NCURSES_EXPORT(int)
slk_wset(int i, const wchar_t *astr, int format)
{
    int result = ERR;
    size_t arglen;
    const wchar_t *str;
    char *mystr;
    mbstate_t state;

    T((T_CALLED("slk_wset(%d, %s, %d)"), i, _nc_viswbuf(astr), format));

    if (astr != 0) {
	init_mb(state);
	str = astr;
	if ((arglen = wcsrtombs(NULL, &str, (size_t) 0, &state)) != (size_t) -1) {
	    if ((mystr = (char *) _nc_doalloc(0, arglen + 1)) != 0) {
		str = astr;
		if (wcsrtombs(mystr, &str, arglen, &state) != (size_t) -1) {
		    /* glibc documentation claims that the terminating L'\0'
		     * is written, but it is not...
		     */
		    mystr[arglen] = 0;
		    result = slk_set(i, mystr, format);
		}
		free(mystr);
	    }
	}
    }
    returnCode(result);
}
