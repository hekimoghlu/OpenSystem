/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
 *  Author: Thomas E. Dickey 1999                                           *
 ****************************************************************************/
/*
 *	trace_tries.c - Tracing/Debugging buffers (keycode tries-trees)
 */

#include <curses.priv.h>

MODULE_ID("$Id: trace_tries.c,v 1.17 2012/10/27 20:50:50 tom Exp $")

#ifdef TRACE
#define my_buffer _nc_globals.tracetry_buf
#define my_length _nc_globals.tracetry_used

static void
recur_tries(TRIES * tree, unsigned level)
{
    if (level > my_length) {
	my_length = (level + 1) * 4;
	my_buffer = (unsigned char *) _nc_doalloc(my_buffer, my_length);
    }

    if (my_buffer != 0) {
	while (tree != 0) {
	    if ((my_buffer[level] = tree->ch) == 0)
		my_buffer[level] = 128;
	    my_buffer[level + 1] = 0;
	    if (tree->value != 0) {
		_tracef("%5d: %s (%s)", tree->value,
			_nc_visbuf((char *) my_buffer), keyname(tree->value));
	    }
	    if (tree->child)
		recur_tries(tree->child, level + 1);
	    tree = tree->sibling;
	}
    }
}

NCURSES_EXPORT(void)
_nc_trace_tries(TRIES * tree)
{
    if ((my_buffer = typeMalloc(unsigned char, my_length = 80)) != 0) {
	_tracef("BEGIN tries %p", (void *) tree);
	recur_tries(tree, 0);
	_tracef(". . . tries %p", (void *) tree);
	free(my_buffer);
    }
}

#else
EMPTY_MODULE(_nc_empty_trace_tries)
#endif
