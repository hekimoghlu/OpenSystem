/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
 *  Author: Thomas E. Dickey                    1999-on                     *
 ****************************************************************************/

/*
 * free_ttype.c -- allocation functions for TERMTYPE
 *
 *	_nc_free_termtype()
 *	use_extended_names()
 *
 */

#include <curses.priv.h>

#include <tic.h>

MODULE_ID("$Id: free_ttype.c,v 1.15 2011/02/06 01:08:31 tom Exp $")

NCURSES_EXPORT(void)
_nc_free_termtype(TERMTYPE *ptr)
{
    T(("_nc_free_termtype(%s)", ptr->term_names));

    FreeIfNeeded(ptr->str_table);
    FreeIfNeeded(ptr->Booleans);
    FreeIfNeeded(ptr->Numbers);
    FreeIfNeeded(ptr->Strings);
#if NCURSES_XNAMES
    FreeIfNeeded(ptr->ext_str_table);
    FreeIfNeeded(ptr->ext_Names);
#endif
    memset(ptr, 0, sizeof(TERMTYPE));
    _nc_free_entry(_nc_head, ptr);
}

#if NCURSES_XNAMES
NCURSES_EXPORT_VAR(bool) _nc_user_definable = TRUE;

NCURSES_EXPORT(int)
use_extended_names(bool flag)
{
    int oldflag = _nc_user_definable;

    START_TRACE();
    T((T_CALLED("use_extended_names(%d)"), flag));
    _nc_user_definable = flag;
    returnBool(oldflag);
}
#endif
