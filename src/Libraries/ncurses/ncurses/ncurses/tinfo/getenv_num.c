/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
 *  Author: Thomas E. Dickey <dickey@clark.net> 1998                        *
 ****************************************************************************/

/*
 *	getenv_num.c -- obtain a number from the environment
 */

#include <curses.priv.h>

MODULE_ID("$Id: getenv_num.c,v 1.6 2013/09/28 20:25:08 tom Exp $")

NCURSES_EXPORT(int)
_nc_getenv_num(const char *name)
{
    char *dst = 0;
    char *src = getenv(name);
    long value;

    if ((src == 0)
	|| (value = strtol(src, &dst, 0)) < 0
	|| (dst == src)
	|| (*dst != '\0')
	|| (int) value < value)
	value = -1;

    return (int) value;
}

NCURSES_EXPORT(void)
_nc_setenv_num(const char *name, int value)
{
    if (name != 0 && value >= 0) {
	char buffer[128];
#if HAVE_SETENV
	_nc_SPRINTF(buffer, _nc_SLIMIT(sizeof(buffer)) "%d", value);
	setenv(name, buffer, 1);
#elif HAVE_PUTENV
	char *s;
	_nc_SPRINTF(buffer, _nc_SLIMIT(sizeof(buffer)) "%s=%d", name, value);
	if ((s = strdup(buffer)) != 0)
	    putenv(s);
#endif
    }
}
