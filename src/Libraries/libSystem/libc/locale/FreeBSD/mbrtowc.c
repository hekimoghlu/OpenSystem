/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/mbrtowc.c,v 1.7 2004/05/12 14:09:04 tjr Exp $");

#include "xlocale_private.h"

#include <wchar.h>
#include "mblocal.h"

size_t
mbrtowc_l(wchar_t * __restrict pwc, const char * __restrict s,
    size_t n, mbstate_t * __restrict ps, locale_t loc)
{
	NORMALIZE_LOCALE(loc);
	if (ps == NULL)
		ps = &loc->__mbs_mbrtowc;
	return (loc->__lc_ctype->__mbrtowc(pwc, s, n, ps, loc));
}

size_t
mbrtowc(wchar_t * __restrict pwc, const char * __restrict s,
    size_t n, mbstate_t * __restrict ps)
{
	return mbrtowc_l(pwc, s, n, ps, __current_locale());
}
