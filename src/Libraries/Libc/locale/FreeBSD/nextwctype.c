/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/nextwctype.c,v 1.1 2004/07/08 06:43:37 tjr Exp $");

#include "xlocale_private.h"

#include <runetype.h>
#include <wchar.h>
#include <wctype.h>

wint_t
nextwctype_l(wint_t wc, wctype_t wct, locale_t loc)
{
	size_t lim;
	_RuneRange *rr;
	_RuneEntry *base, *re;
	int noinc;
	_RuneLocale *rl = XLOCALE_CTYPE(loc)->_CurrentRuneLocale;

	noinc = 0;
	if (wc < _CACHED_RUNES) {
		wc++;
		while (wc < _CACHED_RUNES) {
			if (rl->__runetype[wc] & wct)
				return (wc);
			wc++;
		}
		wc--;
	}
	rr = &rl->__runetype_ext;
	if (rr->__ranges != NULL && wc < rr->__ranges[0].__min) {
		wc = rr->__ranges[0].__min;
		noinc = 1;
	}

	/* Binary search -- see bsearch.c for explanation. */
	base = rr->__ranges;
	for (lim = rr->__nranges; lim != 0; lim >>= 1) {
		re = base + (lim >> 1);
		if (re->__min <= wc && wc <= re->__max)
			goto found;
		else if (wc > re->__max) {
			base = re + 1;
			lim--;
		}
	}
	return (-1);
found:
	if (!noinc)
		wc++;
	if (re->__min <= wc && wc <= re->__max) {
		if (re->__types != NULL) {
			for (; wc <= re->__max; wc++)
				if (re->__types[wc - re->__min] & wct)
					return (wc);
		} else if (re->__map & wct)
			return (wc);
	}
	while (++re < rr->__ranges + rr->__nranges) {
		wc = re->__min;
		if (re->__types != NULL) {
			for (; wc <= re->__max; wc++)
				if (re->__types[wc - re->__min] & wct)
					return (wc);
		} else if (re->__map & wct)
			return (wc);
	}
	return (-1);
}

wint_t
nextwctype(wint_t wc, wctype_t wct)
{
	return nextwctype_l(wc, wct, __current_locale());
}
