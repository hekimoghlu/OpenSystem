/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
 *  Author: Thomas E. Dickey                        1997-on                 *
 *     and: Juergen Pfeifer                         2009                    *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: keyok.c,v 1.14 2014/03/08 20:32:59 tom Exp $")

/*
 * Enable (or disable) ncurses' interpretation of a keycode by adding (or
 * removing) the corresponding 'tries' entry.
 *
 * Do this by storing a second tree of tries, which records the disabled keys. 
 * The simplest way to copy is to make a function that returns the string (with
 * nulls set to 0200), then use that to reinsert the string into the
 * corresponding tree.
 */

NCURSES_EXPORT(int)
NCURSES_SP_NAME(keyok) (NCURSES_SP_DCLx int c, bool flag)
{
    int code = ERR;

    if (HasTerminal(SP_PARM)) {
	T((T_CALLED("keyok(%p, %d,%d)"), (void *) SP_PARM, c, flag));
#ifdef USE_TERM_DRIVER
	code = CallDriver_2(sp, td_kyOk, c, flag);
#else
	T((T_CALLED("keyok(%d,%d)"), c, flag));
	if (c >= 0) {
	    int count = 0;
	    char *s;
	    unsigned ch = (unsigned) c;

	    if (flag) {
		while ((s = _nc_expand_try(SP_PARM->_key_ok,
					   ch, &count, (size_t) 0)) != 0) {
		    if (_nc_remove_key(&(SP_PARM->_key_ok), ch)) {
			code = _nc_add_to_try(&(SP_PARM->_keytry), s, ch);
			free(s);
			count = 0;
			if (code != OK)
			    break;
		    } else {
			free(s);
		    }
		}
	    } else {
		while ((s = _nc_expand_try(SP_PARM->_keytry,
					   ch, &count, (size_t) 0)) != 0) {
		    if (_nc_remove_key(&(SP_PARM->_keytry), ch)) {
			code = _nc_add_to_try(&(SP_PARM->_key_ok), s, ch);
			free(s);
			count = 0;
			if (code != OK)
			    break;
		    } else {
			free(s);
		    }
		}
	    }
	}
#endif
    }
    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
keyok(int c, bool flag)
{
    return NCURSES_SP_NAME(keyok) (CURRENT_SCREEN, c, flag);
}
#endif
