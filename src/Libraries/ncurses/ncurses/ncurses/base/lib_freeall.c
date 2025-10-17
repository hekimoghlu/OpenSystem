/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
 *  Author: Thomas E. Dickey                    1996-on                     *
 ****************************************************************************/

#include <curses.priv.h>
#include <tic.h>

#if HAVE_NC_FREEALL

#if HAVE_LIBDBMALLOC
extern int malloc_errfd;	/* FIXME */
#endif

MODULE_ID("$Id: lib_freeall.c,v 1.63 2015/05/02 23:46:26 tom Exp $")

/*
 * Free all ncurses data.  This is used for testing only (there's no practical
 * use for it as an extension).
 */
NCURSES_EXPORT(void)
NCURSES_SP_NAME(_nc_freeall) (NCURSES_SP_DCL0)
{
    WINDOWLIST *p, *q;
    static va_list empty_va;

    T((T_CALLED("_nc_freeall()")));
#if NO_LEAKS
    if (SP_PARM != 0) {
	if (SP_PARM->_oldnum_list != 0) {
	    FreeAndNull(SP_PARM->_oldnum_list);
	}
	if (SP_PARM->_panelHook.destroy != 0) {
	    SP_PARM->_panelHook.destroy(SP_PARM->_panelHook.stdscr_pseudo_panel);
	}
    }
#endif
    if (SP_PARM != 0) {
	_nc_lock_global(curses);

	while (WindowList(SP_PARM) != 0) {
	    bool deleted = FALSE;

	    /* Delete only windows that're not a parent */
	    for (each_window(SP_PARM, p)) {
		WINDOW *p_win = &(p->win);
		bool found = FALSE;

		for (each_window(SP_PARM, q)) {
		    WINDOW *q_win = &(q->win);
		    if ((p != q)
			&& (q_win->_flags & _SUBWIN)
			&& (p_win == q_win->_parent)) {
			found = TRUE;
			break;
		    }
		}

		if (!found) {
		    if (delwin(p_win) != ERR)
			deleted = TRUE;
		    break;
		}
	    }

	    /*
	     * Don't continue to loop if the list is trashed.
	     */
	    if (!deleted)
		break;
	}
	delscreen(SP_PARM);
	_nc_unlock_global(curses);
    }

    (void) _nc_printf_string(0, empty_va);
#ifdef TRACE
    (void) _nc_trace_buf(-1, (size_t) 0);
#endif
#if USE_WIDEC_SUPPORT
    FreeIfNeeded(_nc_wacs);
#endif
    _nc_leaks_tinfo();

#if HAVE_LIBDBMALLOC
    malloc_dump(malloc_errfd);
#elif HAVE_LIBDMALLOC
#elif HAVE_LIBMPATROL
    __mp_summary();
#elif HAVE_PURIFY
    purify_all_inuse();
#endif
    returnVoid;
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(void)
_nc_freeall(void)
{
    NCURSES_SP_NAME(_nc_freeall) (CURRENT_SCREEN);
}
#endif

NCURSES_EXPORT(void)
NCURSES_SP_NAME(_nc_free_and_exit) (NCURSES_SP_DCLx int code)
{
    NCURSES_SP_NAME(_nc_flush) (NCURSES_SP_ARG);
    NCURSES_SP_NAME(_nc_freeall) (NCURSES_SP_ARG);
#ifdef TRACE
    trace(0);			/* close trace file, freeing its setbuf */
    {
	static va_list fake;
	free(_nc_varargs("?", fake));
    }
#endif
    exit(code);
}

#else
NCURSES_EXPORT(void)
_nc_freeall(void)
{
}

NCURSES_EXPORT(void)
NCURSES_SP_NAME(_nc_free_and_exit) (NCURSES_SP_DCLx int code)
{
    if (SP_PARM) {
	delscreen(SP_PARM);
	if (SP_PARM->_term)
	    NCURSES_SP_NAME(del_curterm) (NCURSES_SP_ARGx SP_PARM->_term);
    }
    exit(code);
}
#endif

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(void)
_nc_free_and_exit(int code)
{
    NCURSES_SP_NAME(_nc_free_and_exit) (CURRENT_SCREEN, code);
}
#endif
