/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
 *	lib_napms.c
 *
 *	The routine napms.
 *
 *	(This file was originally written by Eric Raymond; however except for
 *	comments, none of the original code remains - T.Dickey).
 */

#include <curses.priv.h>

#if HAVE_NANOSLEEP
#include <time.h>
#if HAVE_SYS_TIME_H
#include <sys/time.h>		/* needed for MacOS X DP3 */
#endif
#endif

MODULE_ID("$Id: lib_napms.c,v 1.24 2014/03/08 20:32:59 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(napms) (NCURSES_SP_DCLx int ms)
{
    T((T_CALLED("napms(%d)"), ms));

#ifdef USE_TERM_DRIVER
    if (HasTerminal(SP_PARM)) {
	CallDriver_1(SP_PARM, td_nap, ms);
    }
#else /* !USE_TERM_DRIVER */
#if NCURSES_SP_FUNCS
    (void) sp;
#endif
#if HAVE_NANOSLEEP
    {
	struct timespec request, remaining;
	request.tv_sec = ms / 1000;
	request.tv_nsec = (ms % 1000) * 1000000;
	while (nanosleep(&request, &remaining) == -1
	       && errno == EINTR) {
	    request = remaining;
	}
    }
#else
    _nc_timed_wait(0, 0, ms, (int *) 0 EVENTLIST_2nd(0));
#endif
#endif /* !USE_TERM_DRIVER */

    returnCode(OK);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
napms(int ms)
{
    return NCURSES_SP_NAME(napms) (CURRENT_SCREEN, ms);
}
#endif
