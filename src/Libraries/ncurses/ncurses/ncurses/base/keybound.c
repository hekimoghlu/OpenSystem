/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
 *  Author: Thomas E. Dickey                        1999-on                 *
 *     and: Juergen Pfeifer                         2009                    *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: keybound.c,v 1.11 2011/10/22 16:47:05 tom Exp $")

/*
 * Returns the count'th string definition which is associated with the
 * given keycode.  The result is malloc'd, must be freed by the caller.
 */
NCURSES_EXPORT(char *)
NCURSES_SP_NAME(keybound) (NCURSES_SP_DCLx int code, int count)
{
    char *result = 0;

    T((T_CALLED("keybound(%p, %d,%d)"), (void *) SP_PARM, code, count));
    if (SP_PARM != 0 && code >= 0) {
	result = _nc_expand_try(SP_PARM->_keytry,
				(unsigned) code,
				&count,
				(size_t) 0);
    }
    returnPtr(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char *)
keybound(int code, int count)
{
    return NCURSES_SP_NAME(keybound) (CURRENT_SCREEN, code, count);
}
#endif
