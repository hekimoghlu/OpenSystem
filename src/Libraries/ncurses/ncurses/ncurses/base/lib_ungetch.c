/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 *     and: Thomas E. Dickey                        1996-on                 *
 *     and: Juergen Pfeifer                         2009                    *
 ****************************************************************************/

/*
**	lib_ungetch.c
**
**	The routine ungetch().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_ungetch.c,v 1.16 2012/08/04 17:38:53 tom Exp $")

#include <fifo_defs.h>

#ifdef TRACE
NCURSES_EXPORT(void)
_nc_fifo_dump(SCREEN *sp)
{
    int i;
    T(("head = %d, tail = %d, peek = %d", head, tail, peek));
    for (i = 0; i < 10; i++)
	T(("char %d = %s", i, _nc_tracechar(sp, sp->_fifo[i])));
}
#endif /* TRACE */

NCURSES_EXPORT(int)
safe_ungetch(SCREEN *sp, int ch)
{
    int rc = ERR;

    T((T_CALLED("ungetch(%p,%s)"), (void *) sp, _nc_tracechar(sp, ch)));

#ifdef __APPLE__
    if (!sp) {
	    returnCode(ERR);
    }
#endif

    if (sp != 0 && tail >= 0) {
	if (head < 0) {
	    head = 0;
	    t_inc();
	    peek = tail;	/* no raw keys */
	} else {
	    h_dec();
	}

	sp->_fifo[head] = ch;
	T(("ungetch %s ok", _nc_tracechar(sp, ch)));
#ifdef TRACE
	if (USE_TRACEF(TRACE_IEVENT)) {
	    _nc_fifo_dump(sp);
	    _nc_unlock_global(tracef);
	}
#endif
	rc = OK;
    }
    returnCode(rc);
}

NCURSES_EXPORT(int)
ungetch(int ch)
{
    return safe_ungetch(CURRENT_SCREEN, ch);
}
