/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
 *	lib_tracemse.c - Tracing/Debugging routines (mouse events)
 */

#include <curses.priv.h>

MODULE_ID("$Id: lib_tracemse.c,v 1.22 2014/10/10 09:06:26 tom Exp $")

#ifdef TRACE

#define my_buffer sp->tracemse_buf

NCURSES_EXPORT(char *)
_nc_trace_mmask_t(SCREEN *sp, mmask_t code)
{
#define SHOW(m, s) \
    if ((code & m) == m) { \
	size_t n = strlen(my_buffer); \
	if (n && (my_buffer[n-1] != '{')) \
	_nc_STRCAT(my_buffer, ", ", sizeof(my_buffer)); \
	_nc_STRCAT(my_buffer, s, sizeof(my_buffer)); \
    }

    SHOW(BUTTON1_RELEASED, "release-1");
    SHOW(BUTTON1_PRESSED, "press-1");
    SHOW(BUTTON1_CLICKED, "click-1");
    SHOW(BUTTON1_DOUBLE_CLICKED, "doubleclick-1");
    SHOW(BUTTON1_TRIPLE_CLICKED, "tripleclick-1");
#if NCURSES_MOUSE_VERSION == 1
    SHOW(BUTTON1_RESERVED_EVENT, "reserved-1");
#endif

    SHOW(BUTTON2_RELEASED, "release-2");
    SHOW(BUTTON2_PRESSED, "press-2");
    SHOW(BUTTON2_CLICKED, "click-2");
    SHOW(BUTTON2_DOUBLE_CLICKED, "doubleclick-2");
    SHOW(BUTTON2_TRIPLE_CLICKED, "tripleclick-2");
#if NCURSES_MOUSE_VERSION == 1
    SHOW(BUTTON2_RESERVED_EVENT, "reserved-2");
#endif

    SHOW(BUTTON3_RELEASED, "release-3");
    SHOW(BUTTON3_PRESSED, "press-3");
    SHOW(BUTTON3_CLICKED, "click-3");
    SHOW(BUTTON3_DOUBLE_CLICKED, "doubleclick-3");
    SHOW(BUTTON3_TRIPLE_CLICKED, "tripleclick-3");
#if NCURSES_MOUSE_VERSION == 1
    SHOW(BUTTON3_RESERVED_EVENT, "reserved-3");
#endif

    SHOW(BUTTON4_RELEASED, "release-4");
    SHOW(BUTTON4_PRESSED, "press-4");
    SHOW(BUTTON4_CLICKED, "click-4");
    SHOW(BUTTON4_DOUBLE_CLICKED, "doubleclick-4");
    SHOW(BUTTON4_TRIPLE_CLICKED, "tripleclick-4");
#if NCURSES_MOUSE_VERSION == 1
    SHOW(BUTTON4_RESERVED_EVENT, "reserved-4");
#endif

#if NCURSES_MOUSE_VERSION == 2
    SHOW(BUTTON5_RELEASED, "release-5");
    SHOW(BUTTON5_PRESSED, "press-5");
    SHOW(BUTTON5_CLICKED, "click-5");
    SHOW(BUTTON5_DOUBLE_CLICKED, "doubleclick-5");
    SHOW(BUTTON5_TRIPLE_CLICKED, "tripleclick-5");
#endif

    SHOW(BUTTON_CTRL, "ctrl");
    SHOW(BUTTON_SHIFT, "shift");
    SHOW(BUTTON_ALT, "alt");
    SHOW(ALL_MOUSE_EVENTS, "all-events");
    SHOW(REPORT_MOUSE_POSITION, "position");

#undef SHOW

    if (my_buffer[strlen(my_buffer) - 1] == ' ')
	my_buffer[strlen(my_buffer) - 2] = '\0';

    return (my_buffer);
}

NCURSES_EXPORT(char *)
_nc_tracemouse(SCREEN *sp, MEVENT const *ep)
{
    char *result = 0;

    if (sp != 0) {
	_nc_SPRINTF(my_buffer, _nc_SLIMIT(sizeof(my_buffer))
		    TRACEMSE_FMT,
		    ep->id,
		    ep->x,
		    ep->y,
		    ep->z,
		    (unsigned long) ep->bstate);

	(void) _nc_trace_mmask_t(sp, ep->bstate);
	_nc_STRCAT(my_buffer, "}", sizeof(my_buffer));
	result = (my_buffer);
    }
    return result;
}

NCURSES_EXPORT(mmask_t)
_nc_retrace_mmask_t(SCREEN *sp, mmask_t code)
{
    if (sp != 0) {
	*my_buffer = '\0';
	T((T_RETURN("{%s}"), _nc_trace_mmask_t(sp, code)));
    } else {
	T((T_RETURN("{?}")));
    }
    return code;
}

NCURSES_EXPORT(char *)
_tracemouse(MEVENT const *ep)
{
    return _nc_tracemouse(CURRENT_SCREEN, ep);
}

#else /* !TRACE */
EMPTY_MODULE(_nc_lib_tracemouse)
#endif
