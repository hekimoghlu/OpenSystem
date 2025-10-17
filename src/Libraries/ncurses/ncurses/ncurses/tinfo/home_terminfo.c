/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
 *	home_terminfo.c -- return the $HOME/.terminfo string, expanded
 */

#include <curses.priv.h>
#include <tic.h>

MODULE_ID("$Id: home_terminfo.c,v 1.15 2012/10/27 21:49:14 tom Exp $")

/* ncurses extension...fall back on user's private directory */

#define MyBuffer _nc_globals.home_terminfo

NCURSES_EXPORT(char *)
_nc_home_terminfo(void)
{
    char *result = 0;
#if USE_HOME_TERMINFO
    char *home;

    if (use_terminfo_vars()) {
	if (MyBuffer == 0) {
	    if ((home = getenv("HOME")) != 0) {
		size_t want = (strlen(home) + sizeof(PRIVATE_INFO));
		TYPE_MALLOC(char, want, MyBuffer);
		_nc_SPRINTF(MyBuffer, _nc_SLIMIT(want) PRIVATE_INFO, home);
	    }
	}
	result = MyBuffer;
    }
#endif
    return result;
}
