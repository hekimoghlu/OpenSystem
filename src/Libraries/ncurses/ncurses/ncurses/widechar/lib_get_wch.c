/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
 *  Author: Thomas E. Dickey 2002-on                                        *
 ****************************************************************************/

/*
**	lib_get_wch.c
**
**	The routine get_wch().
**
*/

#include <curses.priv.h>
#include <ctype.h>

MODULE_ID("$Id: lib_get_wch.c,v 1.23 2011/05/28 23:00:29 tom Exp $")

NCURSES_EXPORT(int)
wget_wch(WINDOW *win, wint_t *result)
{
    SCREEN *sp;
    int code;
    char buffer[(MB_LEN_MAX * 9) + 1];	/* allow some redundant shifts */
    int status;
    size_t count = 0;
    int value = 0;
    wchar_t wch;
#ifndef state_unused
    mbstate_t state;
#endif

    T((T_CALLED("wget_wch(%p)"), (void *) win));

    /*
     * We can get a stream of single-byte characters and KEY_xxx codes from
     * _nc_wgetch(), while we want to return a wide character or KEY_xxx code.
     */
    _nc_lock_global(curses);
    sp = _nc_screen_of(win);
    if (sp != 0) {
	for (;;) {
	    T(("reading %d of %d", (int) count + 1, (int) sizeof(buffer)));
	    code = _nc_wgetch(win, &value, TRUE EVENTLIST_2nd((_nc_eventlist
							       *) 0));
	    if (code == ERR) {
		break;
	    } else if (code == KEY_CODE_YES) {
		/*
		 * If we were processing an incomplete multibyte character,
		 * return an error since we have a KEY_xxx code which
		 * interrupts it.  For some cases, we could improve this by
		 * writing a new version of lib_getch.c(!), but it is not clear
		 * whether the improvement would be worth the effort.
		 */
		if (count != 0) {
		    safe_ungetch(SP_PARM, value);
		    code = ERR;
		}
		break;
	    } else if (count + 1 >= sizeof(buffer)) {
		safe_ungetch(SP_PARM, value);
		code = ERR;
		break;
	    } else {
		buffer[count++] = (char) UChar(value);
		reset_mbytes(state);
		status = count_mbytes(buffer, count, state);
		if (status >= 0) {
		    reset_mbytes(state);
		    if (check_mbytes(wch, buffer, count, state) != status) {
			code = ERR;	/* the two calls should match */
			safe_ungetch(SP_PARM, value);
		    }
		    value = wch;
		    break;
		}
	    }
	}
    } else {
	code = ERR;
    }

    if (result != 0)
	*result = (wint_t) value;

    _nc_unlock_global(curses);
    T(("result %#o", value));
    returnCode(code);
}
