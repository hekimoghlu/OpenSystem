/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/string/wcstok.c,v 1.3 2007/12/12 18:33:06 wes Exp $");

#include <wchar.h>

wchar_t *
wcstok(wchar_t * __restrict s, const wchar_t * __restrict delim,
    wchar_t ** __restrict last)
{
	const wchar_t *spanp;
	wchar_t *tok;
	wchar_t c, sc;

	if (s == NULL && (s = *last) == NULL)
		return (NULL);

	/*
	 * Skip (span) leading delimiters (s += wcsspn(s, delim), sort of).
	 */
cont:
	c = *s++;
	for (spanp = delim; (sc = *spanp++) != L'\0';) {
		if (c == sc)
			goto cont;
	}

	if (c == L'\0') {	/* no non-delimiter characters */
		*last = NULL;
		return (NULL);
	}
	tok = s - 1;

	/*
	 * Scan token (scan for delimiters: s += wcscspn(s, delim), sort of).
	 * Note that delim must have one NUL; we stop if we see that, too.
	 */
	for (;;) {
		c = *s++;
		spanp = delim;
		do {
			if ((sc = *spanp++) == c) {
				if (c == L'\0')
					s = NULL;
				else
					s[-1] = L'\0';
				*last = s;
				return (tok);
			}
		} while (sc != L'\0');
	}
	/* NOTREACHED */
}
