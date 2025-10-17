/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
**	Support functions for wide/narrow conversion.
*/

#include <curses.priv.h>

MODULE_ID("$Id: charable.c,v 1.5 2008/07/05 20:51:41 tom Exp $")

NCURSES_EXPORT(bool) _nc_is_charable(wchar_t ch)
{
    bool result;
#if HAVE_WCTOB
    result = (wctob((wint_t) ch) == (int) ch);
#else
    result = (_nc_to_char(ch) >= 0);
#endif
    return result;
}

NCURSES_EXPORT(int) _nc_to_char(wint_t ch)
{
    int result;
#if HAVE_WCTOB
    result = wctob(ch);
#elif HAVE_WCTOMB
    char temp[MB_LEN_MAX];
    result = wctomb(temp, ch);
    if (strlen(temp) == 1)
	result = UChar(temp[0]);
    else
	result = -1;
#endif
    return result;
}

NCURSES_EXPORT(wint_t) _nc_to_widechar(int ch)
{
    wint_t result;
#if HAVE_BTOWC
    result = btowc(ch);
#elif HAVE_MBTOWC
    wchar_t convert;
    char temp[2];
    temp[0] = ch;
    temp[1] = '\0';
    if (mbtowc(&convert, temp, 1) >= 0)
	result = convert;
    else
	result = WEOF;
#endif
    return result;
}
