/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#if 0
__FBSDID("FreeBSD: src/lib/libc/string/strxfrm.c,v 1.15 2002/09/06 11:24:06 tjr Exp ");
#endif
__FBSDID("$FreeBSD: src/lib/libc/string/wcsxfrm.c,v 1.3 2004/04/07 09:47:56 tjr Exp $");

#include "xlocale_private.h"

#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <errno.h>
#include "collate.h"

#define WCS_XFRM_OFFSET	1

size_t
wcsxfrm_l(wchar_t * __restrict dest, const wchar_t * __restrict src, size_t len,
    locale_t loc)
{
	size_t slen;
	wchar_t *xf[COLL_WEIGHTS_MAX];
	int sverrno;

	if (*src == L'\0') {
		if (len != 0)
			*dest = L'\0';
		return (0);
	}

	NORMALIZE_LOCALE(loc);
	if (XLOCALE_COLLATE(loc)->__collate_load_error) {
		slen = wcslen(src);
		if (len > 0) {
			if (slen < len)
				wcscpy(dest, src);
			else {
				wcsncpy(dest, src, len - 1);
				dest[len - 1] = L'\0';
			}
		}
		return (slen);
	}

	__collate_xfrm(src, xf, loc);

	slen = wcslen(xf[0]);
	if (xf[1])
		slen += wcslen(xf[1]) + 1;
	if (len > 0) {
		wchar_t *w = xf[0];
		while (len > 1) {
			if (!*w)
				break;
			*dest++ = *w++ + WCS_XFRM_OFFSET;
			len--;
		}
		if ((w = xf[1]) != NULL) {
			if (len > 1)
				*dest++ = WCS_XFRM_OFFSET;
			while (len > 1) {
				if (!*w)
					break;
				*dest++ = *w++ + WCS_XFRM_OFFSET;
				len--;
			}
		}
		*dest = 0;
 	}
	sverrno = errno;
	free(xf[0]);
	free(xf[1]);
	errno = sverrno;
 
	return (slen);
}

size_t
wcsxfrm(wchar_t * __restrict dest, const wchar_t * __restrict src, size_t len)
{
	return wcsxfrm_l(dest, src, len, __current_locale());
}
