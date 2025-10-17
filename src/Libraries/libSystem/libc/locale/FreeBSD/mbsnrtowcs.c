/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/mbsnrtowcs.c,v 1.1 2004/07/21 10:54:57 tjr Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <wchar.h>
#include "mblocal.h"

size_t
mbsnrtowcs_l(wchar_t * __restrict dst, const char ** __restrict src,
    size_t nms, size_t len, mbstate_t * __restrict ps, locale_t loc)
{
	NORMALIZE_LOCALE(loc);
	if (ps == NULL)
		ps = &loc->__mbs_mbsnrtowcs;
	return (loc->__lc_ctype->__mbsnrtowcs(dst, src, nms, len, ps, loc));
}

size_t
mbsnrtowcs(wchar_t * __restrict dst, const char ** __restrict src,
    size_t nms, size_t len, mbstate_t * __restrict ps)
{
	return mbsnrtowcs_l(dst, src, nms, len, ps, __current_locale());
}

__private_extern__ size_t
__mbsnrtowcs_std(wchar_t * __restrict dst, const char ** __restrict src,
    size_t nms, size_t len, mbstate_t * __restrict ps, locale_t loc)
{
	const char *s;
	size_t nchr;
	wchar_t wc;
	size_t nb;
	size_t (*__mbrtowc)(wchar_t * __restrict, const char * __restrict,
	    size_t, mbstate_t * __restrict, locale_t)
	    = loc->__lc_ctype->__mbrtowc;

	s = *src;
	nchr = 0;

	if (dst == NULL) {
		for (;;) {
			if ((nb = __mbrtowc(&wc, s, nms, ps, loc)) == (size_t)-1)
				/* Invalid sequence - mbrtowc() sets errno. */
				return ((size_t)-1);
			else if (nb == 0 || nb == (size_t)-2)
				return (nchr);
			s += nb;
			nms -= nb;
			nchr++;
		}
		/*NOTREACHED*/
	}

	while (len-- > 0) {
		if ((nb = __mbrtowc(dst, s, nms, ps, loc)) == (size_t)-1) {
			*src = s;
			return ((size_t)-1);
		} else if (nb == (size_t)-2) {
			*src = s + nms;
			return (nchr);
		} else if (nb == 0) {
			*src = NULL;
			return (nchr);
		}
		s += nb;
		nms -= nb;
		nchr++;
		dst++;
	}
	*src = s;
	return (nchr);
}
