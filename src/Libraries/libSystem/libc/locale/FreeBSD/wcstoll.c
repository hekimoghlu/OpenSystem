/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include <sys/cdefs.h>
#if 0
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)strtoq.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
__FBSDID("FreeBSD: src/lib/libc/stdlib/strtoll.c,v 1.19 2002/09/06 11:23:59 tjr Exp ");
#endif
__FBSDID("$FreeBSD: src/lib/libc/locale/wcstoll.c,v 1.2 2007/01/09 00:28:01 imp Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <wchar.h>
#include <wctype.h>

/*
 * Convert a wide character string to a long long integer.
 */
long long
wcstoll_l(const wchar_t * __restrict nptr, wchar_t ** __restrict endptr,
    int base, locale_t loc)
{
	const wchar_t *s;
	unsigned long long acc;
	wchar_t c;
	unsigned long long cutoff;
	int neg, any, cutlim;

	NORMALIZE_LOCALE(loc);
	/*
	 * See strtoll for comments as to the logic used.
	 */
	s = nptr;
	do {
		c = *s++;
	} while (iswspace_l(c, loc));
	if (c == L'-') {
		neg = 1;
		c = *s++;
	} else {
		neg = 0;
		if (c == L'+')
			c = *s++;
	}
	if ((base == 0 || base == 16) &&
	    c == L'0' && (*s == L'x' || *s == L'X')) {
		c = s[1];
		s += 2;
		base = 16;
	}
	if (base == 0)
		base = c == L'0' ? 8 : 10;
	acc = any = 0;
	if (base < 2 || base > 36)
		goto noconv;

	cutoff = neg ? (unsigned long long)-(LLONG_MIN + LLONG_MAX) + LLONG_MAX
	    : LLONG_MAX;
	cutlim = cutoff % base;
	cutoff /= base;
	for ( ; ; c = *s++) {
#ifdef notyet
		if (iswdigit_l(c, loc))
			c = digittoint_l(c, loc);
		else
#endif
		if (c >= L'0' && c <= L'9')
			c -= L'0';
		else if (c >= L'A' && c <= L'Z')
			c -= L'A' - 10;
		else if (c >= L'a' && c <= L'z')
			c -= L'a' - 10;
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
		acc = neg ? LLONG_MIN : LLONG_MAX;
		errno = ERANGE;
	} else if (!any) {
noconv:
		errno = EINVAL;
	} else if (neg)
		acc = -acc;
	if (endptr != NULL)
		*endptr = (wchar_t *)(any ? s - 1 : nptr);
	return (acc);
}

long long
wcstoll(const wchar_t * __restrict nptr, wchar_t ** __restrict endptr, int base)
{
	return wcstoll_l(nptr, endptr, base, __current_locale());
}
