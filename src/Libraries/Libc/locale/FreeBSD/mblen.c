/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/mblen.c,v 1.9 2004/07/29 06:18:40 tjr Exp $");

#include "xlocale_private.h"

#include <stdlib.h>
#include <wchar.h>
#include "mblocal.h"

int
mblen_l(const char *s, size_t n, locale_t loc)
{
	static const mbstate_t initial;
	size_t rval;

	NORMALIZE_LOCALE(loc);
	if (s == NULL) {
		/* No support for state dependent encodings. */
		loc->__mbs_mblen = initial;
		return (0);
	}
	rval = XLOCALE_CTYPE(loc)->__mbrtowc(NULL, s, n, &loc->__mbs_mblen, loc);
	if (rval == (size_t)-1 || rval == (size_t)-2)
		return (-1);
	return ((int)rval);
}

int
mblen(const char *s, size_t n)
{
	return mblen_l(s, n, __current_locale());
}
