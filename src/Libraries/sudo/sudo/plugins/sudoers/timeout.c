/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>

#include "sudo_compat.h"
#include "sudoers_debug.h"
#include "parse.h"

/*
 * Parse a command timeout in sudoers in the format 1d2h3m4s
 * (days, hours, minutes, seconds) or a number of seconds with no suffix.
 * Returns the number of seconds or -1 on error.
 */
int
parse_timeout(const char *timestr)
{
    debug_decl(parse_timeout, SUDOERS_DEBUG_PARSER);
    const char suffixes[] = "dhms";
    const char *cp = timestr;
    int timeout = 0;
    int idx = 0;

    do {
	char *ep;
	char ch;
	long l;

	/* Parse number, must be present and positive. */
	errno = 0;
	l = strtol(cp, &ep, 10);
	if (ep == cp) {
	    /* missing timeout */
	    errno = EINVAL;
	    debug_return_int(-1);
	}
	if (errno == ERANGE || l < 0 || l > INT_MAX)
	    goto overflow;

	/* Find a matching suffix or return an error. */
	if (*ep != '\0') {
	    ch = tolower((unsigned char)*ep++);
	    while (suffixes[idx] != ch) {
		if (suffixes[idx] == '\0') {
		    /* parse error */
		    errno = EINVAL;
		    debug_return_int(-1);
		}
		idx++;
	    }

	    /* Apply suffix. */
	    switch (ch) {
	    case 'd':
		if (l > INT_MAX / (24 * 60 * 60))
		    goto overflow;
		l *= 24 * 60 * 60;
		break;
	    case 'h':
		if (l > INT_MAX / (60 * 60))
		    goto overflow;
		l *= 60 * 60;
		break;
	    case 'm':
		if (l > INT_MAX / 60)
		    goto overflow;
		l *= 60;
		break;
	    }
	    if (l > INT_MAX - timeout)
		goto overflow;
	}
	cp = ep;

	timeout += l;
    } while (*cp != '\0');

    debug_return_int(timeout);
overflow:
    errno = ERANGE;
    debug_return_int(-1);
}
