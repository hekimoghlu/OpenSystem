/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
 ****************************************************************************/

/*
**	lib_scanw.c
**
**	The routines scanw(), wscanw() and friends.
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_scanw.c,v 1.13 2011/10/22 16:31:35 tom Exp $")

NCURSES_EXPORT(int)
vwscanw(WINDOW *win, NCURSES_CONST char *fmt, va_list argp)
{
    char buf[BUFSIZ];

    if (wgetnstr(win, buf, (int) sizeof(buf) - 1) == ERR)
	return (ERR);

    return (vsscanf(buf, fmt, argp));
}

NCURSES_EXPORT(int)
scanw(NCURSES_CONST char *fmt,...)
{
    int code;
    va_list ap;

    T(("scanw(\"%s\",...) called", fmt));

    va_start(ap, fmt);
    code = vwscanw(stdscr, fmt, ap);
    va_end(ap);
    return (code);
}

NCURSES_EXPORT(int)
wscanw(WINDOW *win, NCURSES_CONST char *fmt,...)
{
    int code;
    va_list ap;

    T(("wscanw(%p,\"%s\",...) called", (void *) win, fmt));

    va_start(ap, fmt);
    code = vwscanw(win, fmt, ap);
    va_end(ap);
    return (code);
}

NCURSES_EXPORT(int)
mvscanw(int y, int x, NCURSES_CONST char *fmt,...)
{
    int code;
    va_list ap;

    va_start(ap, fmt);
    code = (move(y, x) == OK) ? vwscanw(stdscr, fmt, ap) : ERR;
    va_end(ap);
    return (code);
}

NCURSES_EXPORT(int)
mvwscanw(WINDOW *win, int y, int x, NCURSES_CONST char *fmt,...)
{
    int code;
    va_list ap;

    va_start(ap, fmt);
    code = (wmove(win, y, x) == OK) ? vwscanw(win, fmt, ap) : ERR;
    va_end(ap);
    return (code);
}
