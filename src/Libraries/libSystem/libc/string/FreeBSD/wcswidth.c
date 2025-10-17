/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/string/wcswidth.c,v 1.7 2007/01/09 00:28:12 imp Exp $");

#include "xlocale_private.h"

#include <wchar.h>

int
wcswidth_l(const wchar_t *pwcs, size_t n, locale_t loc)
{
	wchar_t wc;
	int len, l;

	NORMALIZE_LOCALE(loc);
	len = 0;
	while (n-- > 0 && (wc = *pwcs++) != L'\0') {
		if ((l = wcwidth_l(wc, loc)) < 0)
			return (-1);
		len += l;
	}
	return (len);
}


int
wcswidth(const wchar_t *pwcs, size_t n)
{
	return wcswidth_l(pwcs, n, __current_locale());
}

