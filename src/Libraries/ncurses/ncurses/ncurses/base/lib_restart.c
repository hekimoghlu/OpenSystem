/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
 *     and: Juergen Pfeifer                         2008                    *
 ****************************************************************************/

/*
 * Terminfo-only terminal setup routines:
 *
 *		int restartterm(const char *, int, int *)
 */

#include <curses.priv.h>

MODULE_ID("$Id: lib_restart.c,v 1.16 2015/06/27 18:12:15 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(restartterm) (NCURSES_SP_DCLx
			      NCURSES_CONST char *termp,
			      int filenum,
			      int *errret)
{
    int result;
#ifdef USE_TERM_DRIVER
    TERMINAL *new_term = 0;
#endif

    START_TRACE();
    T((T_CALLED("restartterm(%p,%s,%d,%p)"),
       (void *) SP_PARM,
       termp,
       filenum,
       (void *) errret));

    if (TINFO_SETUP_TERM(&new_term, termp, filenum, errret, FALSE) != OK) {
	result = ERR;
    } else if (SP_PARM != 0) {
	int saveecho = SP_PARM->_echo;
	int savecbreak = SP_PARM->_cbreak;
	int saveraw = SP_PARM->_raw;
	int savenl = SP_PARM->_nl;

#ifdef USE_TERM_DRIVER
	SP_PARM->_term = new_term;
#endif
	if (saveecho) {
	    NCURSES_SP_NAME(echo) (NCURSES_SP_ARG);
	} else {
	    NCURSES_SP_NAME(noecho) (NCURSES_SP_ARG);
	}

	if (savecbreak) {
	    NCURSES_SP_NAME(cbreak) (NCURSES_SP_ARG);
	    NCURSES_SP_NAME(noraw) (NCURSES_SP_ARG);
	} else if (saveraw) {
	    NCURSES_SP_NAME(nocbreak) (NCURSES_SP_ARG);
	    NCURSES_SP_NAME(raw) (NCURSES_SP_ARG);
	} else {
	    NCURSES_SP_NAME(nocbreak) (NCURSES_SP_ARG);
	    NCURSES_SP_NAME(noraw) (NCURSES_SP_ARG);
	}
	if (savenl) {
	    NCURSES_SP_NAME(nl) (NCURSES_SP_ARG);
	} else {
	    NCURSES_SP_NAME(nonl) (NCURSES_SP_ARG);
	}

	NCURSES_SP_NAME(reset_prog_mode) (NCURSES_SP_ARG);

#if USE_SIZECHANGE
	_nc_update_screensize(SP_PARM);
#endif

	result = OK;
    } else {
	result = ERR;
    }
    returnCode(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
restartterm(NCURSES_CONST char *termp, int filenum, int *errret)
{
    START_TRACE();
    return NCURSES_SP_NAME(restartterm) (CURRENT_SCREEN, termp, filenum, errret);
}
#endif
