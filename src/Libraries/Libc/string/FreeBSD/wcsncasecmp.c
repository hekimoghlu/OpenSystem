/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/string/wcsncasecmp.c,v 1.1 2009/02/28 06:00:58 das Exp $");

#include "xlocale_private.h"

#include <wchar.h>
#include <wctype.h>

int
wcsncasecmp_l(const wchar_t *s1, const wchar_t *s2, size_t n, locale_t loc)
{
	wchar_t c1, c2;

	if (n == 0)
		return (0);
	for (; *s1; s1++, s2++) {
		c1 = towlower_l(*s1, loc);
		c2 = towlower_l(*s2, loc);
		if (c1 != c2)
			return ((int)c1 - c2);
		if (--n == 0)
			return (0);
	}
	return (-*s2);
}

int
wcsncasecmp(const wchar_t *s1, const wchar_t *s2, size_t n) {
	return wcsncasecmp_l(s1, s2, n, __current_locale());
}

