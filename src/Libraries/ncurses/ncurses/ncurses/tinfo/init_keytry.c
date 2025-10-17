/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include <curses.priv.h>
#include <tic.h>		/* struct tinfo_fkeys */

MODULE_ID("$Id: init_keytry.c,v 1.17 2010/04/24 22:29:56 tom Exp $")

/*
**      _nc_init_keytry()
**
**      Construct the try for the current terminal's keypad keys.
**
*/

/*
 * Internal entrypoints use SCREEN* parameter to obtain capabilities rather
 * than cur_term.
 */
#undef CUR
#define CUR SP_TERMTYPE

#if	BROKEN_LINKER
#undef	_nc_tinfo_fkeys
#endif

/* LINT_PREPRO
#if 0*/
#include <init_keytry.h>
/* LINT_PREPRO
#endif*/

#if	BROKEN_LINKER
const struct tinfo_fkeys *
_nc_tinfo_fkeysf(void)
{
    return _nc_tinfo_fkeys;
}
#endif

NCURSES_EXPORT(void)
_nc_init_keytry(SCREEN *sp)
{
    unsigned n;

    /* The sp->_keytry value is initialized in newterm(), where the sp
     * structure is created, because we can not tell where keypad() or
     * mouse_activate() (which will call keyok()) are first called.
     */

    if (sp != 0) {
	for (n = 0; _nc_tinfo_fkeys[n].code; n++) {
	    if (_nc_tinfo_fkeys[n].offset < STRCOUNT) {
		(void) _nc_add_to_try(&(sp->_keytry),
				      CUR Strings[_nc_tinfo_fkeys[n].offset],
				      _nc_tinfo_fkeys[n].code);
	    }
	}
#if NCURSES_XNAMES
	/*
	 * Add any of the extended strings to the tries if their name begins
	 * with 'k', i.e., they follow the convention of other terminfo key
	 * names.
	 */
	{
	    TERMTYPE *tp = &(sp->_term->type);
	    for (n = STRCOUNT; n < NUM_STRINGS(tp); ++n) {
		const char *name = ExtStrname(tp, (int) n, strnames);
		char *value = tp->Strings[n];
		if (name != 0
		    && *name == 'k'
		    && value != 0
		    && NCURSES_SP_NAME(key_defined) (NCURSES_SP_ARGx
						     value) == 0) {
		    (void) _nc_add_to_try(&(sp->_keytry),
					  value,
					  n - STRCOUNT + KEY_MAX);
		}
	    }
	}
#endif
#ifdef TRACE
	_nc_trace_tries(sp->_keytry);
#endif
    }
}
