/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wctob.c,v 1.4 2004/05/12 14:26:54 tjr Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <stdio.h>
#include <wchar.h>
#include "mblocal.h"

int
wctob_l(wint_t c, locale_t loc)
{
	static const mbstate_t initial;
	mbstate_t mbs = initial;
	char buf[MB_LEN_MAX];

	NORMALIZE_LOCALE(loc);
	if (c == WEOF || loc->__lc_ctype->__wcrtomb(buf, c, &mbs, loc) != 1)
		return (EOF);
	return ((unsigned char)*buf);
}

int
wctob(wint_t c)
{
	return wctob_l(c, __current_locale());
}
