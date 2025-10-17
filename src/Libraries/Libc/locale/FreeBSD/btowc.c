/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/btowc.c,v 1.4 2004/05/12 14:26:54 tjr Exp $");

#include "xlocale_private.h"

#include <stdio.h>
#include <wchar.h>
#include "mblocal.h"

wint_t
btowc_l(int c, locale_t loc)
{
	static const mbstate_t initial;
	mbstate_t mbs = initial;
	char cc;
	wchar_t wc;

	NORMALIZE_LOCALE(loc);
	if (c == EOF)
		return (WEOF);
	/*
	 * We expect mbrtowc() to return 0 or 1, hence the check for n > 1
	 * which detects error return values as well as "impossible" byte
	 * counts.
	 */
	cc = (char)c;
	if (XLOCALE_CTYPE(loc)->__mbrtowc(&wc, &cc, 1, &mbs, loc) > 1)
		return (WEOF);
	return (wc);
}

wint_t
btowc(int c)
{
	return btowc_l(c, __current_locale());
}
