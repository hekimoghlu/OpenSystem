/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#include <curses.priv.h>

MODULE_ID("$Id: lib_erasewchar.c,v 1.3 2014/02/23 01:21:08 tom Exp $")

/*
 *	erasewchar()
 *
 *	Return erase character as given in cur_term->Ottyb.
 *
 */

NCURSES_EXPORT(int)
erasewchar(wchar_t *wch)
{
    int value;
    int result = ERR;

    T((T_CALLED("erasewchar()")));
    if ((value = erasechar()) != ERR) {
	*wch = (wchar_t) value;
	result = OK;
    }
    returnCode(result);
}

/*
 *	killwchar()
 *
 *	Return kill character as given in cur_term->Ottyb.
 *
 */

NCURSES_EXPORT(int)
killwchar(wchar_t *wch)
{
    int value;
    int result = ERR;

    T((T_CALLED("killwchar()")));
    if ((value = killchar()) != ERR) {
	*wch = (wchar_t) value;
	result = OK;
    }
    returnCode(result);
}
