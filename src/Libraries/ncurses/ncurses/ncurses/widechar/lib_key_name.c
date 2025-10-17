/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
**	lib_key_name.c
**
**	The routine key_name().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_key_name.c,v 1.3 2008/10/11 20:15:14 tom Exp $")

NCURSES_EXPORT(NCURSES_CONST char *)
key_name(wchar_t c)
{
    cchar_t my_cchar;
    wchar_t *my_wchars;
    size_t len;

    /* FIXME: move to _nc_globals */
    static char result[MB_LEN_MAX + 1];

    memset(&my_cchar, 0, sizeof(my_cchar));
    my_cchar.chars[0] = c;
    my_cchar.chars[1] = L'\0';

    my_wchars = wunctrl(&my_cchar);
    len = wcstombs(result, my_wchars, sizeof(result) - 1);
    if (isEILSEQ(len) || (len == 0)) {
	return 0;
    }

    result[len] = '\0';
    return result;
}
