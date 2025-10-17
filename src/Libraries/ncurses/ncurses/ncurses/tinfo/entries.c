/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
 *  Author: Thomas E. Dickey                                                *
 *     and: Juergen Pfeifer                                                 *
 ****************************************************************************/

#include <curses.priv.h>

#include <ctype.h>

#include <tic.h>

MODULE_ID("$Id: entries.c,v 1.21 2012/05/05 20:33:44 tom Exp $")

/****************************************************************************
 *
 * Entry queue handling
 *
 ****************************************************************************/
/*
 *  The entry list is a doubly linked list with NULLs terminating the lists:
 *
 *	  ---------   ---------   ---------
 *	  |       |   |       |   |       |   offset
 *        |-------|   |-------|   |-------|
 *	  |   ----+-->|   ----+-->|  NULL |   next
 *	  |-------|   |-------|   |-------|
 *	  |  NULL |<--+----   |<--+----   |   last
 *	  ---------   ---------   ---------
 *	      ^                       ^
 *	      |                       |
 *	      |                       |
 *	   _nc_head                _nc_tail
 */

NCURSES_EXPORT_VAR(ENTRY *) _nc_head = 0;
NCURSES_EXPORT_VAR(ENTRY *) _nc_tail = 0;

NCURSES_EXPORT(void)
_nc_free_entry(ENTRY * headp, TERMTYPE *tterm)
/* free the allocated storage consumed by the given list entry */
{
    ENTRY *ep;

    if ((ep = _nc_delink_entry(headp, tterm)) != 0) {
	free(ep);
    }
}

NCURSES_EXPORT(void)
_nc_free_entries(ENTRY * headp)
/* free the allocated storage consumed by list entries */
{
    (void) headp;		/* unused - _nc_head is altered here! */

    while (_nc_head != 0) {
	_nc_free_termtype(&(_nc_head->tterm));
    }
}

NCURSES_EXPORT(ENTRY *)
_nc_delink_entry(ENTRY * headp, TERMTYPE *tterm)
/* delink the allocated storage for the given list entry */
{
    ENTRY *ep, *last;

    for (last = 0, ep = headp; ep != 0; last = ep, ep = ep->next) {
	if (&(ep->tterm) == tterm) {
	    if (last != 0) {
		last->next = ep->next;
	    }
	    if (ep->next != 0) {
		ep->next->last = last;
	    }
	    if (ep == _nc_head) {
		_nc_head = ep->next;
	    }
	    if (ep == _nc_tail) {
		_nc_tail = last;
	    }
	    break;
	}
    }
    return ep;
}

NCURSES_EXPORT(void)
_nc_leaks_tinfo(void)
{
#if NO_LEAKS
    char *s;
#endif

    T((T_CALLED("_nc_free_tinfo()")));
#if NO_LEAKS
    _nc_free_tparm();
    _nc_tgetent_leaks();

    if (TerminalOf(CURRENT_SCREEN) != 0) {
	del_curterm(TerminalOf(CURRENT_SCREEN));
    }

    _nc_comp_captab_leaks();
    _nc_free_entries(_nc_head);
    _nc_get_type(0);
    _nc_first_name(0);
    _nc_db_iterator_leaks();
    _nc_keyname_leaks();
#if BROKEN_LINKER || USE_REENTRANT
    _nc_names_leaks();
    _nc_codes_leaks();
    FreeIfNeeded(_nc_prescreen.real_acs_map);
#endif
    _nc_comp_error_leaks();

    if ((s = _nc_home_terminfo()) != 0)
	free(s);

#ifdef TRACE
    trace(0);
    _nc_trace_buf(-1, (size_t) 0);
#endif

#endif /* NO_LEAKS */
    returnVoid;
}

#if NO_LEAKS
NCURSES_EXPORT(void)
_nc_free_tinfo(int code)
{
    _nc_leaks_tinfo();
    exit(code);
}
#endif
