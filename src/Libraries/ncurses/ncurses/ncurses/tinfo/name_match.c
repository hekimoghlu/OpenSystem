/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
 *  Author: Thomas E. Dickey                    1999-on                     *
 ****************************************************************************/

#include <curses.priv.h>
#include <tic.h>

MODULE_ID("$Id: name_match.c,v 1.23 2013/05/25 20:20:08 tom Exp $")

#define FirstName _nc_globals.first_name

#if NCURSES_USE_TERMCAP && NCURSES_XNAMES
static const char *
skip_index(const char *name)
{
    if ((_nc_syntax == SYN_TERMCAP) && _nc_user_definable) {
	const char *bar = strchr(name, '|');
	if (bar != 0 && (bar - name) == 2)
	    name = bar + 1;
    }
    return name;
}
#endif

/*
 * Get the primary name from the given name list.  For terminfo, this is the
 * first name.  For termcap, this may be the second name, if the first one
 * happens to be two characters.
 */
NCURSES_EXPORT(char *)
_nc_first_name(const char *const sp)
{
    unsigned n;

#if NO_LEAKS
    if (sp == 0) {
	if (FirstName != 0) {
	    FreeAndNull(FirstName);
	}
    } else
#endif
    {
	if (FirstName == 0)
	    FirstName = typeMalloc(char, MAX_NAME_SIZE + 1);

	if (FirstName != 0) {
	    const char *src = sp;
#if NCURSES_USE_TERMCAP && NCURSES_XNAMES
	    src = skip_index(sp);
#endif
	    for (n = 0; n < MAX_NAME_SIZE; n++) {
		if ((FirstName[n] = src[n]) == '\0'
		    || (FirstName[n] == '|'))
		    break;
	    }
	    FirstName[n] = '\0';
	}
    }
    return (FirstName);
}

/*
 * Is the given name matched in namelist?
 */
NCURSES_EXPORT(int)
_nc_name_match(const char *const namelst, const char *const name, const char *const delim)
{
    const char *s, *d, *t;
    int code, found;

    if ((s = namelst) != 0) {
	while (*s != '\0') {
	    for (d = name; *d != '\0'; d++) {
		if (*s != *d)
		    break;
		s++;
	    }
	    found = FALSE;
	    for (code = TRUE; *s != '\0'; code = FALSE, s++) {
		for (t = delim; *t != '\0'; t++) {
		    if (*s == *t) {
			found = TRUE;
			break;
		    }
		}
		if (found)
		    break;
	    }
	    if (code && *d == '\0')
		return code;
	    if (*s++ == 0)
		break;
	}
    }
    return FALSE;
}
