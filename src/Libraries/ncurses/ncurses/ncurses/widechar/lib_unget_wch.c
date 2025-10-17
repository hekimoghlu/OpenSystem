/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
 *  Author: Thomas E. Dickey 2002                                           *
 ****************************************************************************/

/*
**	lib_unget_wch.c
**
**	The routine unget_wch().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_unget_wch.c,v 1.15 2011/10/22 16:34:50 tom Exp $")

/*
 * Wrapper for wcrtomb() which obtains the length needed for the given
 * wide-character 'source'.
 */
NCURSES_EXPORT(size_t)
_nc_wcrtomb(char *target, wchar_t source, mbstate_t * state)
{
    int result;

    if (target == 0) {
	wchar_t temp[2];
	const wchar_t *tempp = temp;
	temp[0] = source;
	temp[1] = 0;
	result = (int) wcsrtombs(NULL, &tempp, (size_t) 0, state);
    } else {
	result = (int) wcrtomb(target, source, state);
    }
    if (!isEILSEQ(result) && (result == 0))
	result = 1;
    return (size_t) result;
}

NCURSES_EXPORT(int)
NCURSES_SP_NAME(unget_wch) (NCURSES_SP_DCLx const wchar_t wch)
{
    int result = OK;
    mbstate_t state;
    size_t length;
    int n;

    T((T_CALLED("unget_wch(%p, %#lx)"), (void *) SP_PARM, (unsigned long) wch));

    init_mb(state);
    length = _nc_wcrtomb(0, wch, &state);

    if (length != (size_t) (-1)
	&& length != 0) {
	char *string;

	if ((string = (char *) malloc(length)) != 0) {
	    init_mb(state);
	    /* ignore the result, since we already validated the character */
	    IGNORE_RC((int) wcrtomb(string, wch, &state));

	    for (n = (int) (length - 1); n >= 0; --n) {
		if (NCURSES_SP_NAME(ungetch) (NCURSES_SP_ARGx
					      UChar(string[n])) !=OK) {
		    result = ERR;
		    break;
		}
	    }
	    free(string);
	} else {
	    result = ERR;
	}
    } else {
	result = ERR;
    }

    returnCode(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
unget_wch(const wchar_t wch)
{
    return NCURSES_SP_NAME(unget_wch) (CURRENT_SCREEN, wch);
}
#endif
