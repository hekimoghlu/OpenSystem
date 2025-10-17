/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/tolower.c,v 1.13 2007/01/09 00:28:01 imp Exp $");
  
#include "xlocale_private.h"

#include <ctype.h>
#include <stdio.h>
#include <runetype.h>

__ct_rune_t
___tolower_l(c, loc)
	__ct_rune_t c;
	locale_t loc;
{
	size_t lim;
	_RuneRange *rr;
	_RuneEntry *base, *re;

	if (c < 0 || c == EOF)
		return(c);

	NORMALIZE_LOCALE(loc);
	/*
	 * the following is not used by tolower(), but can be used by
	 * tolower_l().  This provides the oppurtunity to optimize tolower()
	 * when compatibility for Panther and lower is no longer needed
	 */
	if (c < _CACHED_RUNES)
		return loc->__lc_ctype->_CurrentRuneLocale.__maplower[c];
	rr = &loc->__lc_ctype->_CurrentRuneLocale.__maplower_ext;
	/* Binary search -- see bsearch.c for explanation. */
	base = rr->__ranges;
	for (lim = rr->__nranges; lim != 0; lim >>= 1) {
		re = base + (lim >> 1);
		if (re->__min <= c && c <= re->__max)
			return (re->__map + c - re->__min);
		else if (c > re->__max) {
			base = re + 1;
			lim--;
		}
	}

	return(c);
}

__ct_rune_t
___tolower(c)
	__ct_rune_t c;
{
	return ___tolower_l(c, __current_locale());
}
