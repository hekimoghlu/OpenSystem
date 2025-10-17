/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wcstombs.c,v 1.11 2009/01/15 18:53:52 rdivacky Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <stdlib.h>
#include <wchar.h>
#include "mblocal.h"

size_t
wcstombs_l(char * __restrict s, const wchar_t * __restrict pwcs, size_t n,
    locale_t loc)
{
	static const mbstate_t initial;
	mbstate_t mbs;
	const wchar_t *pwcsp;

	NORMALIZE_LOCALE(loc);
	mbs = initial;
	pwcsp = pwcs;
	return (XLOCALE_CTYPE(loc)->__wcsnrtombs(s, &pwcsp, SIZE_T_MAX, n, &mbs, loc));
}

size_t
wcstombs(char * __restrict s, const wchar_t * __restrict pwcs, size_t n)
{
	return wcstombs_l(s, pwcs, n, __current_locale());
}
