/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wcsrtombs.c,v 1.6 2004/07/21 10:54:57 tjr Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "mblocal.h"

size_t
wcsrtombs_l(char * __restrict dst, const wchar_t ** __restrict src, size_t len,
    mbstate_t * __restrict ps, locale_t loc)
{
	NORMALIZE_LOCALE(loc);
	if (ps == NULL)
		ps = &loc->__mbs_wcsrtombs;
	return (XLOCALE_CTYPE(loc)->__wcsnrtombs(dst, src, SIZE_T_MAX, len, ps, loc));
}

size_t
wcsrtombs(char * __restrict dst, const wchar_t ** __restrict src, size_t len,
    mbstate_t * __restrict ps)
{
	return wcsrtombs_l(dst, src, len, ps, __current_locale());
}
