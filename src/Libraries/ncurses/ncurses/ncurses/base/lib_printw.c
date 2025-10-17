/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
 *  Author: Thomas E. Dickey            1997-on                             *
 ****************************************************************************/

/*
**	lib_printw.c
**
**	The routines printw(), wprintw() and friends.
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_printw.c,v 1.23 2012/09/03 17:55:28 tom Exp $")

NCURSES_EXPORT(int)
printw(const char *fmt,...)
{
    va_list argp;
    int code;

#ifdef TRACE
    va_list argq;
    va_start(argq, fmt);
    T((T_CALLED("printw(%s%s)"),
       _nc_visbuf(fmt), _nc_varargs(fmt, argq)));
    va_end(argq);
#endif

    va_start(argp, fmt);
    code = vwprintw(stdscr, fmt, argp);
    va_end(argp);

    returnCode(code);
}

NCURSES_EXPORT(int)
wprintw(WINDOW *win, const char *fmt,...)
{
    va_list argp;
    int code;

#ifdef TRACE
    va_list argq;
    va_start(argq, fmt);
    T((T_CALLED("wprintw(%p,%s%s)"),
       (void *) win, _nc_visbuf(fmt), _nc_varargs(fmt, argq)));
    va_end(argq);
#endif

    va_start(argp, fmt);
    code = vwprintw(win, fmt, argp);
    va_end(argp);

    returnCode(code);
}

NCURSES_EXPORT(int)
mvprintw(int y, int x, const char *fmt,...)
{
    va_list argp;
    int code;

#ifdef TRACE
    va_list argq;
    va_start(argq, fmt);
    T((T_CALLED("mvprintw(%d,%d,%s%s)"),
       y, x, _nc_visbuf(fmt), _nc_varargs(fmt, argq)));
    va_end(argq);
#endif

    if ((code = move(y, x)) != ERR) {
	va_start(argp, fmt);
	code = vwprintw(stdscr, fmt, argp);
	va_end(argp);
    }
    returnCode(code);
}

NCURSES_EXPORT(int)
mvwprintw(WINDOW *win, int y, int x, const char *fmt,...)
{
    va_list argp;
    int code;

#ifdef TRACE
    va_list argq;
    va_start(argq, fmt);
    T((T_CALLED("mvwprintw(%d,%d,%p,%s%s)"),
       y, x, (void *) win, _nc_visbuf(fmt), _nc_varargs(fmt, argq)));
    va_end(argq);
#endif

    if ((code = wmove(win, y, x)) != ERR) {
	va_start(argp, fmt);
	code = vwprintw(win, fmt, argp);
	va_end(argp);
    }
    returnCode(code);
}

NCURSES_EXPORT(int)
vwprintw(WINDOW *win, const char *fmt, va_list argp)
{
    char *buf;
    int code = ERR;
#if NCURSES_SP_FUNCS
    SCREEN *sp = _nc_screen_of(win);
#endif

    T((T_CALLED("vwprintw(%p,%s,va_list)"), (void *) win, _nc_visbuf(fmt)));

    buf = NCURSES_SP_NAME(_nc_printf_string) (NCURSES_SP_ARGx fmt, argp);
    if (buf != 0) {
	code = waddstr(win, buf);
    }
    returnCode(code);
}
