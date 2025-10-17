/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
__FBSDID("$FreeBSD$");

#include <errno.h>
#include <limits.h>
#include <runetype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "mblocal.h"

static size_t	_ascii_mbrtowc(wchar_t * __restrict, const char * __restrict,
		    size_t, mbstate_t * __restrict, locale_t);
static int	_ascii_mbsinit(const mbstate_t *, locale_t);
static size_t	_ascii_mbsnrtowcs(wchar_t * __restrict dst,
		    const char ** __restrict src, size_t nms, size_t len,
		    mbstate_t * __restrict ps __unused, locale_t);
static size_t	_ascii_wcrtomb(char * __restrict, wchar_t,
		    mbstate_t * __restrict, locale_t);
static size_t	_ascii_wcsnrtombs(char * __restrict, const wchar_t ** __restrict,
		    size_t, size_t, mbstate_t * __restrict, locale_t);

int
_ascii_init(struct __xlocale_st_runelocale *xrl)
{

	xrl->__mbrtowc = _ascii_mbrtowc;
	xrl->__mbsinit = _ascii_mbsinit;
	xrl->__mbsnrtowcs = _ascii_mbsnrtowcs;
	xrl->__wcrtomb = _ascii_wcrtomb;
	xrl->__wcsnrtombs = _ascii_wcsnrtombs;
	xrl->__mb_cur_max = 1;
	xrl->__mb_sb_limit = 128;
	return(0);
}

static int
_ascii_mbsinit(const mbstate_t *ps __unused, locale_t loc __unused)
{

	/*
	 * Encoding is not state dependent - we are always in the
	 * initial state.
	 */
	return (1);
}

static size_t
_ascii_mbrtowc(wchar_t * __restrict pwc, const char * __restrict s, size_t n,
    mbstate_t * __restrict ps __unused, locale_t loc __unused)
{

	if (s == NULL)
		/* Reset to initial shift state (no-op) */
		return (0);
	if (n == 0)
		/* Incomplete multibyte sequence */
		return ((size_t)-2);
	if (*s & 0x80) {
		errno = EILSEQ;
		return ((size_t)-1);
	}
	if (pwc != NULL)
		*pwc = (unsigned char)*s;
	return (*s == '\0' ? 0 : 1);
}

static size_t
_ascii_wcrtomb(char * __restrict s, wchar_t wc,
    mbstate_t * __restrict ps __unused, locale_t loc __unused)
{

	if (s == NULL)
		/* Reset to initial shift state (no-op) */
		return (1);
	if (wc < 0 || wc > 127) {
		errno = EILSEQ;
		return ((size_t)-1);
	}
	*s = (unsigned char)wc;
	return (1);
}

static size_t
_ascii_mbsnrtowcs(wchar_t * __restrict dst, const char ** __restrict src,
    size_t nms, size_t len, mbstate_t * __restrict ps __unused, locale_t loc __unused)
{
	const char *s;
	size_t nchr;

	if (dst == NULL) {
		for (s = *src; nms > 0 && *s != '\0'; s++, nms--) {
			if (*s & 0x80) {
				errno = EILSEQ;
				return ((size_t)-1);
			}
		}
		return (s - *src);
	}

	s = *src;
	nchr = 0;
	while (len-- > 0 && nms-- > 0) {
		if (*s & 0x80) {
			*src = s;
			errno = EILSEQ;
			return ((size_t)-1);
		}
		if ((*dst++ = (unsigned char)*s++) == L'\0') {
			*src = NULL;
			return (nchr);
		}
		nchr++;
	}
	*src = s;
	return (nchr);
}

static size_t
_ascii_wcsnrtombs(char * __restrict dst, const wchar_t ** __restrict src,
    size_t nwc, size_t len, mbstate_t * __restrict ps __unused, locale_t loc __unused)
{
	const wchar_t *s;
	size_t nchr;

	if (dst == NULL) {
		for (s = *src; nwc > 0 && *s != L'\0'; s++, nwc--) {
			if (*s < 0 || *s > 127) {
				errno = EILSEQ;
				return ((size_t)-1);
			}
		}
		return (s - *src);
	}

	s = *src;
	nchr = 0;
	while (len-- > 0 && nwc-- > 0) {
		if (*s < 0 || *s > 127) {
			*src = s;
			errno = EILSEQ;
			return ((size_t)-1);
		}
		if ((*dst++ = *s++) == '\0') {
			*src = NULL;
			return (nchr);
		}
		nchr++;
	}
	*src = s;
	return (nchr);
}

