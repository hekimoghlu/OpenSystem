/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "from @(#)strtoul.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/strtoumax.c,v 1.11 2007/01/09 00:28:10 imp Exp $");

#include "xlocale_private.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <inttypes.h>

/*
 * Convert a string to a uintmax_t integer.
 *
 * Assumes that the upper and lower case
 * alphabets and digits are each contiguous.
 */
uintmax_t
strtoumax_l(const char * __restrict nptr, char ** __restrict endptr, int base,
    locale_t loc)
{
	const char *s;
	uintmax_t acc;
	char c;
	uintmax_t cutoff;
	int neg, any, cutlim;

	NORMALIZE_LOCALE(loc);
	/*
	 * See strtoimax for comments as to the logic used.
	 */
	s = nptr;
	do {
		c = *s++;
	} while (isspace_l((unsigned char)c, loc));
	if (c == '-') {
		neg = 1;
		c = *s++;
	} else {
		neg = 0;
		if (c == '+')
			c = *s++;
	}
	if ((base == 0 || base == 16) &&
	    c == '0' && (*s == 'x' || *s == 'X') &&
	    ((s[1] >= '0' && s[1] <= '9') ||
	    (s[1] >= 'A' && s[1] <= 'F') ||
	    (s[1] >= 'a' && s[1] <= 'f'))) {
		c = s[1];
		s += 2;
		base = 16;
	}
	if (base == 0)
		base = c == '0' ? 8 : 10;
	acc = any = 0;
	if (base < 2 || base > 36)
		goto noconv;

	cutoff = UINTMAX_MAX / base;
	cutlim = UINTMAX_MAX % base;
	for ( ; ; c = *s++) {
		if (c >= '0' && c <= '9')
			c -= '0';
		else if (c >= 'A' && c <= 'Z')
			c -= 'A' - 10;
		else if (c >= 'a' && c <= 'z')
			c -= 'a' - 10;
		else
			break;
		if (c >= base)
			break;
		if (any < 0 || acc > cutoff || (acc == cutoff && c > cutlim))
			any = -1;
		else {
			any = 1;
			acc *= base;
			acc += c;
		}
	}
	if (any < 0) {
		acc = UINTMAX_MAX;
		errno = ERANGE;
	} else if (!any) {
noconv:
		errno = EINVAL;
	} else if (neg)
		acc = -acc;
	if (endptr != NULL)
		*endptr = (char *)(any ? s - 1 : nptr);
	return (acc);
}

uintmax_t
strtoumax(const char * __restrict nptr, char ** __restrict endptr, int base)
{
	return strtoumax_l(nptr, endptr, base, __current_locale());
}
