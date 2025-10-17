/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wcsnrtombs.c,v 1.3 2005/02/12 08:45:12 stefanf Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "mblocal.h"

size_t
wcsnrtombs_l(char * __restrict dst, const wchar_t ** __restrict src, size_t nwc,
    size_t len, mbstate_t * __restrict ps, locale_t loc)
{
	NORMALIZE_LOCALE(loc);
	if (ps == NULL)
		ps = &loc->__mbs_wcsnrtombs;
	return (XLOCALE_CTYPE(loc)->__wcsnrtombs(dst, src, nwc, len, ps, loc));
}

size_t
wcsnrtombs(char * __restrict dst, const wchar_t ** __restrict src, size_t nwc,
    size_t len, mbstate_t * __restrict ps)
{
	return wcsnrtombs_l(dst, src, nwc, len, ps, __current_locale());
}

__private_extern__ size_t
__wcsnrtombs_std(char * __restrict dst, const wchar_t ** __restrict src,
    size_t nwc, size_t len, mbstate_t * __restrict ps, locale_t loc)
{
	mbstate_t mbsbak;
	char buf[MB_LEN_MAX];
	const wchar_t *s;
	size_t nbytes;
	size_t nb;
	struct xlocale_ctype *runeLocale = XLOCALE_CTYPE(loc);
	size_t (*__wcrtomb)(char * __restrict, wchar_t, mbstate_t * __restrict, locale_t) = runeLocale->__wcrtomb;
	int mb_cur_max = runeLocale->__mb_cur_max;

	s = *src;
	nbytes = 0;

	if (dst == NULL) {
		while (nwc-- > 0) {
			if ((nb = __wcrtomb(buf, *s, ps, loc)) == (size_t)-1)
				/* Invalid character - wcrtomb() sets errno. */
				return ((size_t)-1);
			else if (*s == L'\0')
				return (nbytes + nb - 1);
			s++;
			nbytes += nb;
		}
		return (nbytes);
	}

	while (len > 0 && nwc-- > 0) {
		if (len > (size_t)mb_cur_max) {
			/* Enough space to translate in-place. */
			if ((nb = __wcrtomb(dst, *s, ps, loc)) == (size_t)-1) {
				*src = s;
				return ((size_t)-1);
			}
		} else {
			/*
			 * May not be enough space; use temp. buffer.
			 *
			 * We need to save a copy of the conversion state
			 * here so we can restore it if the multibyte
			 * character is too long for the buffer.
			 */
			mbsbak = *ps;
			if ((nb = __wcrtomb(buf, *s, ps, loc)) == (size_t)-1) {
				*src = s;
				return ((size_t)-1);
			}
			if (nb > (int)len) {
				/* MB sequence for character won't fit. */
				*ps = mbsbak;
				break;
			}
			memcpy(dst, buf, nb);
		}
		if (*s == L'\0') {
			*src = NULL;
			return (nbytes + nb - 1);
		}
		s++;
		dst += nb;
		len -= nb;
		nbytes += nb;
	}
	*src = s;
	return (nbytes);
}
