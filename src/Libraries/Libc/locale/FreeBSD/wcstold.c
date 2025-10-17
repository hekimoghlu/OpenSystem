/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wcstold.c,v 1.4 2004/04/07 09:47:56 tjr Exp $");

#include "xlocale_private.h"

#include <stdlib.h>
#include <wchar.h>
#include <wctype.h>
#include <_simple.h>

/*
 * See wcstod() for comments as to the logic used.
 */

extern size_t __wcs_end_offset(const char * __restrict buf, const char * __restrict end, locale_t loc);

long double
wcstold_l(const wchar_t * __restrict nptr, wchar_t ** __restrict endptr,
    locale_t loc)
{
	static const mbstate_t initial;
	mbstate_t mbs;
	long double val;
	char *buf, *end;
	size_t len;
	locale_t ctype;
	_SIMPLE_STRING b;
	char mb[MB_CUR_MAX + 1];
	const wchar_t *nptr0 = nptr;
	const wchar_t *first;

	NORMALIZE_LOCALE(loc);
	ctype = __numeric_ctype(loc);

	while (iswspace_l(*nptr, ctype))
		nptr++;

	if ((b = _simple_salloc()) == NULL)
		return (0.0);

	first = nptr;
	mbs = initial;
	while (*nptr && (len = wcrtomb_l(mb, *nptr, &mbs, ctype)) != (size_t)-1) {
		mb[len] = 0;
		if (_simple_sappend(b, mb) < 0) { /* no memory */
			_simple_sfree(b);
			return (0.0);
		}
		nptr++;
	}

	buf = _simple_string(b);
	val = strtold_l(buf, &end, loc);

	if (endptr != NULL)
		*endptr = (end == buf) ? (wchar_t *)nptr0 : ((wchar_t *)first + __wcs_end_offset(buf, end, loc));

	_simple_sfree(b);

	return (val);
}

long double
wcstold(const wchar_t * __restrict nptr, wchar_t ** __restrict endptr)
{
	return wcstold_l(nptr, endptr, __current_locale());
}
