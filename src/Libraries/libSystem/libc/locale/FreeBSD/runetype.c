/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/runetype.c,v 1.14 2007/01/09 00:28:00 imp Exp $");

#include "xlocale_private.h"

#include <ctype.h>
#include <stdio.h>
#include <runetype.h>

unsigned long
___runetype_l(__ct_rune_t c, locale_t loc)
{
	size_t lim;
	_RuneRange *rr;
	_RuneEntry *base, *re;

	if (c < 0 || c == EOF)
		return(0L);

	NORMALIZE_LOCALE(loc);
	rr = &loc->__lc_ctype->_CurrentRuneLocale.__runetype_ext;
	/* Binary search -- see bsearch.c for explanation. */
	base = rr->__ranges;
	for (lim = rr->__nranges; lim != 0; lim >>= 1) {
		re = base + (lim >> 1);
		if (re->__min <= c && c <= re->__max) {
			if (re->__types)
			    return(re->__types[c - re->__min]);
			else
			    return(re->__map);
		} else if (c > re->__max) {
			base = re + 1;
			lim--;
		}
	}

	return(0L);
}

unsigned long
___runetype(__ct_rune_t c)
{
	return ___runetype_l(c, __current_locale());
}
