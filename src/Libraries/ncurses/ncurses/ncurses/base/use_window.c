/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
 *     Author: Thomas E. Dickey                        2007                 *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: use_window.c,v 1.9 2009/10/24 22:40:24 tom Exp $")

NCURSES_EXPORT(int)
use_window(WINDOW *win, NCURSES_WINDOW_CB func, void *data)
{
    int code = OK;

    T((T_CALLED("use_window(%p,%p,%p)"), (void *) win, func, data));
    _nc_lock_global(curses);
    code = func(win, data);
    _nc_unlock_global(curses);

    returnCode(code);
}
