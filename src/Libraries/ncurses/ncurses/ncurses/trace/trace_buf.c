/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
 *  Author: Thomas E. Dickey                 1997-on                        *
 ****************************************************************************/
/*
 *	trace_buf.c - Tracing/Debugging buffers (attributes)
 */

#include <curses.priv.h>

MODULE_ID("$Id: trace_buf.c,v 1.20 2012/02/22 22:34:31 tom Exp $")

#ifdef TRACE

#define MyList _nc_globals.tracebuf_ptr
#define MySize _nc_globals.tracebuf_used

static char *
_nc_trace_alloc(int bufnum, size_t want)
{
    char *result = 0;

    if (bufnum >= 0) {
	if ((size_t) (bufnum + 1) > MySize) {
	    size_t need = (size_t) (bufnum + 1) * 2;
	    if ((MyList = typeRealloc(TRACEBUF, need, MyList)) != 0) {
		while (need > MySize)
		    MyList[MySize++].text = 0;
	    }
	}

	if (MyList != 0) {
	    if (MyList[bufnum].text == 0
		|| want > MyList[bufnum].size) {
		MyList[bufnum].text = typeRealloc(char, want, MyList[bufnum].text);
		if (MyList[bufnum].text != 0)
		    MyList[bufnum].size = want;
	    }
	    result = MyList[bufnum].text;
	}
    }
#if NO_LEAKS
    else {
	if (MySize) {
	    if (MyList) {
		while (MySize--) {
		    if (MyList[MySize].text != 0) {
			free(MyList[MySize].text);
		    }
		}
		free(MyList);
		MyList = 0;
	    }
	    MySize = 0;
	}
    }
#endif
    return result;
}

/*
 * (re)Allocate a buffer big enough for the caller's wants.
 */
NCURSES_EXPORT(char *)
_nc_trace_buf(int bufnum, size_t want)
{
    char *result = _nc_trace_alloc(bufnum, want);
    if (result != 0)
	*result = '\0';
    return result;
}

/*
 * Append a new string to an existing buffer.
 */
NCURSES_EXPORT(char *)
_nc_trace_bufcat(int bufnum, const char *value)
{
    char *buffer = _nc_trace_alloc(bufnum, (size_t) 0);
    if (buffer != 0) {
	size_t have = strlen(buffer);
	size_t need = strlen(value) + have;

	buffer = _nc_trace_alloc(bufnum, 1 + need);
	if (buffer != 0)
	    _nc_STRCPY(buffer + have, value, need);

    }
    return buffer;
}
#else
EMPTY_MODULE(_nc_empty_trace_buf)
#endif /* TRACE */
