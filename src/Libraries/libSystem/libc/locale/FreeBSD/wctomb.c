/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wctomb.c,v 1.8 2004/07/29 06:18:40 tjr Exp $");

#include "xlocale_private.h"

#include <stdlib.h>
#include <wchar.h>
#include "mblocal.h"

int
wctomb_l(char *s, wchar_t wchar, locale_t loc)
{
	static const mbstate_t initial;
	size_t rval;

	NORMALIZE_LOCALE(loc);
	if (s == NULL) {
		/* No support for state dependent encodings. */
		loc->__mbs_wctomb = initial;
		return (0);
	}
	if ((rval = loc->__lc_ctype->__wcrtomb(s, wchar, &loc->__mbs_wctomb, loc)) == (size_t)-1)
		return (-1);
	return ((int)rval);
}

int
wctomb(char *s, wchar_t wchar)
{
	return wctomb_l(s, wchar, __current_locale());
}
