/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
 *  Author: Thomas E. Dickey <dickey@clark.net> 1997                        *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: nc_panel.c,v 1.5 2009/04/11 21:05:10 tom Exp $")

NCURSES_EXPORT(struct panelhook *)
NCURSES_SP_NAME(_nc_panelhook) (NCURSES_SP_DCL0)
{
    return (SP_PARM
	    ? &(SP_PARM->_panelHook)
	    : (CURRENT_SCREEN
	       ? &(CURRENT_SCREEN->_panelHook)
	       : 0));
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(struct panelhook *)
_nc_panelhook(void)
{
    return NCURSES_SP_NAME(_nc_panelhook) (CURRENT_SCREEN);
}
#endif
