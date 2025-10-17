/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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
 *     and: Thomas E. Dickey                        2002                    *
 *     and: Juergen Pfeifer                         2009                    *
 ****************************************************************************/

/*
 *	lib_kernel.c
 *
 *	Misc. low-level routines:
 *		erasechar()
 *		killchar()
 *		flushinp()
 *
 * The baudrate() and delay_output() functions could logically live here,
 * but are in other modules to reduce the static-link size of programs
 * that use only these facilities.
 */

#include <curses.priv.h>

MODULE_ID("$Id: lib_kernel.c,v 1.31 2010/12/19 01:21:19 tom Exp $")

static int
_nc_vdisable(void)
{
    int value = -1;
#if defined(_POSIX_VDISABLE) && HAVE_UNISTD_H
    value = _POSIX_VDISABLE;
#endif
#if defined(_PC_VDISABLE)
    if (value == -1) {
	value = (int) fpathconf(0, _PC_VDISABLE);
	if (value == -1) {
	    value = 0377;
	}
    }
#elif defined(VDISABLE)
    if (value == -1)
	value = VDISABLE;
#endif
    return value;
}

/*
 *	erasechar()
 *
 *	Return erase character as given in cur_term->Ottyb.
 *
 */

NCURSES_EXPORT(char)
NCURSES_SP_NAME(erasechar) (NCURSES_SP_DCL0)
{
    int result = ERR;
    TERMINAL *termp = TerminalOf(SP_PARM);

    T((T_CALLED("erasechar(%p)"), (void *) SP_PARM));

    if (termp != 0) {
#ifdef TERMIOS
	result = termp->Ottyb.c_cc[VERASE];
	if (result == _nc_vdisable())
	    result = ERR;
#else
	result = termp->Ottyb.sg_erase;
#endif
    }
    returnChar((char) result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char)
erasechar(void)
{
    return NCURSES_SP_NAME(erasechar) (CURRENT_SCREEN);
}
#endif

/*
 *	killchar()
 *
 *	Return kill character as given in cur_term->Ottyb.
 *
 */

NCURSES_EXPORT(char)
NCURSES_SP_NAME(killchar) (NCURSES_SP_DCL0)
{
    int result = ERR;
    TERMINAL *termp = TerminalOf(SP_PARM);

    T((T_CALLED("killchar(%p)"), (void *) SP_PARM));

    if (termp != 0) {
#ifdef TERMIOS
	result = termp->Ottyb.c_cc[VKILL];
	if (result == _nc_vdisable())
	    result = ERR;
#else
	result = termp->Ottyb.sg_kill;
#endif
    }
    returnChar((char) result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char)
killchar(void)
{
    return NCURSES_SP_NAME(killchar) (CURRENT_SCREEN);
}
#endif

/*
 *	flushinp()
 *
 *	Flush any input on cur_term->Filedes
 *
 */

NCURSES_EXPORT(int)
NCURSES_SP_NAME(flushinp) (NCURSES_SP_DCL0)
{
    TERMINAL *termp = TerminalOf(SP_PARM);

    T((T_CALLED("flushinp(%p)"), (void *) SP_PARM));

    if (termp != 0) {
#ifdef TERMIOS
	tcflush(termp->Filedes, TCIFLUSH);
#else
	errno = 0;
	do {
	    ioctl(termp->Filedes, TIOCFLUSH, 0);
	} while
	    (errno == EINTR);
#endif
	if (SP_PARM) {
	    SP_PARM->_fifohead = -1;
	    SP_PARM->_fifotail = 0;
	    SP_PARM->_fifopeek = 0;
	}
	returnCode(OK);
    }
    returnCode(ERR);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
flushinp(void)
{
    return NCURSES_SP_NAME(flushinp) (CURRENT_SCREEN);
}
#endif
