/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
 ****************************************************************************/

/*
 *	lib_tracechr.c - Tracing/Debugging routines
 */
#include <curses.priv.h>

#include <ctype.h>

MODULE_ID("$Id: lib_tracechr.c,v 1.22 2012/02/22 22:40:24 tom Exp $")

#ifdef TRACE

#define MyBufSize sizeof(_nc_globals.tracechr_buf)

NCURSES_EXPORT(char *)
_nc_tracechar(SCREEN *sp, int ch)
{
    NCURSES_CONST char *name;
    char *MyBuffer = ((sp != 0)
		      ? sp->tracechr_buf
		      : _nc_globals.tracechr_buf);

    if (ch > KEY_MIN || ch < 0) {
	name = safe_keyname(SP_PARM, ch);
	if (name == 0 || *name == '\0')
	    name = "NULL";
	_nc_SPRINTF(MyBuffer, _nc_SLIMIT(MyBufSize)
		    "'%.30s' = %#03o", name, ch);
    } else if (!is8bits(ch) || !isprint(UChar(ch))) {
	/*
	 * workaround for glibc bug:
	 * sprintf changes the result from unctrl() to an empty string if it
	 * does not correspond to a valid multibyte sequence.
	 */
	_nc_SPRINTF(MyBuffer, _nc_SLIMIT(MyBufSize)
		    "%#03o", ch);
    } else {
	name = safe_unctrl(SP_PARM, (chtype) ch);
	if (name == 0 || *name == 0)
	    name = "null";	/* shouldn't happen */
	_nc_SPRINTF(MyBuffer, _nc_SLIMIT(MyBufSize)
		    "'%.30s' = %#03o", name, ch);
    }
    return (MyBuffer);
}

NCURSES_EXPORT(char *)
_tracechar(int ch)
{
    return _nc_tracechar(CURRENT_SCREEN, ch);
}
#else
EMPTY_MODULE(_nc_lib_tracechr)
#endif
