/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wcrtomb.c,v 1.8 2004/05/12 14:09:04 tjr Exp $");

#include "xlocale_private.h"

#include <wchar.h>
#include "mblocal.h"

size_t
wcrtomb_l(char * __restrict s, wchar_t wc, mbstate_t * __restrict ps,
    locale_t loc)
{
	NORMALIZE_LOCALE(loc);
	if (ps == NULL)
		ps = &loc->__mbs_wcrtomb;
	return (XLOCALE_CTYPE(loc)->__wcrtomb(s, wc, ps, loc));
}

size_t
wcrtomb(char * __restrict s, wchar_t wc, mbstate_t * __restrict ps)
{
	return wcrtomb_l(s, wc, ps, __current_locale());
}
