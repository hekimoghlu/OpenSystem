/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
 *  Author: Thomas E. Dickey          2005                                  *
 *          Juergen Pfeifer           2009                                  *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: legacy_coding.c,v 1.5 2009/10/24 22:15:00 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(use_legacy_coding) (NCURSES_SP_DCLx int level)
{
    int result = ERR;

    T((T_CALLED("use_legacy_coding(%p,%d)"), (void *) SP_PARM, level));
    if (level >= 0 && level <= 2 && SP_PARM != 0) {
	result = SP_PARM->_legacy_coding;
	SP_PARM->_legacy_coding = level;
    }
    returnCode(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
use_legacy_coding(int level)
{
    return NCURSES_SP_NAME(use_legacy_coding) (CURRENT_SCREEN, level);
}
#endif
