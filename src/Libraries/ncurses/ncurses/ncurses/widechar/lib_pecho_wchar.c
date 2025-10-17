/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#include <curses.priv.h>

MODULE_ID("$Id: lib_pecho_wchar.c,v 1.2 2009/10/24 22:43:32 tom Exp $")

NCURSES_EXPORT(int)
pecho_wchar(WINDOW *pad, const cchar_t *wch)
{
    T((T_CALLED("pecho_wchar(%p, %s)"), (void *) pad, _tracech_t(wch)));

    if (pad == 0)
	returnCode(ERR);

    if (!(pad->_flags & _ISPAD))
	returnCode(wecho_wchar(pad, wch));

    wadd_wch(pad, wch);
    prefresh(pad, pad->_pad._pad_y,
	     pad->_pad._pad_x,
	     pad->_pad._pad_top,
	     pad->_pad._pad_left,
	     pad->_pad._pad_bottom,
	     pad->_pad._pad_right);

    returnCode(OK);
}
