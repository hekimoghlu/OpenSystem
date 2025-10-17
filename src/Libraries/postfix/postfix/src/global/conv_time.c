/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
/* System library. */

#include <sys_defs.h>
#include <limits.h>			/* INT_MAX */
#include <stdlib.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>

/* Global library. */

#include <conv_time.h>

#define MINUTE	(60)
#define HOUR	(60 * MINUTE)
#define DAY	(24 * HOUR)
#define WEEK	(7 * DAY)

/* conv_time - convert time value */

int     conv_time(const char *strval, int *timval, int def_unit)
{
    char   *end;
    int     intval;
    long    longval;

    errno = 0;
    intval = longval = strtol(strval, &end, 10);
    if (*strval == 0 || errno == ERANGE || longval != intval || intval < 0
	/* || (*end != 0 && end[1] != 0) */)
	return (0);

    switch (*end ? *end : def_unit) {
    case 'w':
	if (intval < INT_MAX / WEEK) {
	    *timval = intval * WEEK;
	    return (1);
	} else {
	    return (0);
	}
    case 'd':
	if (intval < INT_MAX / DAY) {
	    *timval = intval * DAY;
	    return (1);
	} else {
	    return (0);
	}
    case 'h':
	if (intval < INT_MAX / HOUR) {
	    *timval = intval * HOUR;
	    return (1);
	} else {
	    return (0);
	}
    case 'm':
	if (intval < INT_MAX / MINUTE) {
	    *timval = intval * MINUTE;
	    return (1);
	} else {
	    return (0);
	}
    case 's':
	*timval = intval;
	return (1);
    }
    return (0);
}
