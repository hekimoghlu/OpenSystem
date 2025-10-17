/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include <ctype.h>
#include "tclInt.h"

/*
 *----------------------------------------------------------------------
 *
 * strtol --
 *
 *	Convert an ASCII string into an integer.
 *
 * Results:
 *	The return value is the integer equivalent of string. If endPtr is
 *	non-NULL, then *endPtr is filled in with the character after the last
 *	one that was part of the integer. If string doesn't contain a valid
 *	integer value, then zero is returned and *endPtr is set to string.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

long int
strtol(
    CONST char *string,		/* String of ASCII digits, possibly preceded
				 * by white space. For bases greater than 10,
				 * either lower- or upper-case digits may be
				 * used. */
    char **endPtr,		/* Where to store address of terminating
				 * character, or NULL. */
    int base)			/* Base for conversion. Must be less than 37.
				 * If 0, then the base is chosen from the
				 * leading characters of string: "0x" means
				 * hex, "0" means octal, anything else means
				 * decimal. */
{
    register CONST char *p;
    long result;

    /*
     * Skip any leading blanks.
     */

    p = string;
    while (isspace(UCHAR(*p))) {
	p += 1;
    }

    /*
     * Check for a sign.
     */

    if (*p == '-') {
	p += 1;
	result = -(strtoul(p, endPtr, base));
    } else {
	if (*p == '+') {
	    p += 1;
	}
	result = strtoul(p, endPtr, base);
    }
    if ((result == 0) && (endPtr != 0) && (*endPtr == p)) {
	*endPtr = (char *) string;
    }
    return result;
}
