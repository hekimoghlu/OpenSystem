/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)mystring.c	8.2 (Berkeley) 5/4/95";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: head/bin/sh/mystring.c 326025 2017-11-20 19:49:47Z pfg $");

/*
 * String functions.
 *
 *	equal(s1, s2)		Return true if strings are equal.
 *	number(s)		Convert a string of digits to an integer.
 *	is_number(s)		Return true if s is a string of digits.
 */

#include <stdlib.h>
#include "shell.h"
#include "syntax.h"
#include "error.h"
#include "mystring.h"


char nullstr[1];		/* zero length string */

/*
 * equal - #defined in mystring.h
 */


/*
 * Convert a string of digits to an integer, printing an error message on
 * failure.
 */

int
number(const char *s)
{
	if (! is_number(s))
		error("Illegal number: %s", s);
	return atoi(s);
}



/*
 * Check for a valid number.  This should be elsewhere.
 */

int
is_number(const char *p)
{
	const char *q;

	if (*p == '\0')
		return 0;
	while (*p == '0')
		p++;
	for (q = p; *q != '\0'; q++)
		if (! is_digit(*q))
			return 0;
	if (q - p > 10 ||
	    (q - p == 10 && memcmp(p, "2147483647", 10) > 0))
		return 0;
	return 1;
}
