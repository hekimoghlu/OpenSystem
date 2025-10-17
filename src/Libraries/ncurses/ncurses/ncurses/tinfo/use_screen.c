/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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

MODULE_ID("$Id: use_screen.c,v 1.8 2009/10/24 22:40:20 tom Exp $")

NCURSES_EXPORT(int)
use_screen(SCREEN *screen, NCURSES_SCREEN_CB func, void *data)
{
    SCREEN *save_SP;
    int code = OK;

    T((T_CALLED("use_screen(%p,%p,%p)"), (void *) screen, func, (void *) data));

    /*
     * FIXME - add a flag so a given thread can check if _it_ has already
     * recurred through this point, return an error if so.
     */
    _nc_lock_global(curses);
    save_SP = CURRENT_SCREEN;
    set_term(screen);

    code = func(screen, data);

    set_term(save_SP);
    _nc_unlock_global(curses);
    returnCode(code);
}
