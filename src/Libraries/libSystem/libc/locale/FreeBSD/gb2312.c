/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#include <sys/param.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/gb2312.c,v 1.10 2007/10/13 16:28:21 ache Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <runetype.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "mblocal.h"

#define GB2312_MB_CUR_MAX	2

static size_t	_GB2312_mbrtowc(wchar_t * __restrict, const char * __restrict,
		    size_t, mbstate_t * __restrict, locale_t);
static int	_GB2312_mbsinit(const mbstate_t *, locale_t);
static size_t	_GB2312_wcrtomb(char * __restrict, wchar_t,
		    mbstate_t * __restrict, locale_t);
typedef struct {
	int	count;
	u_char	bytes[2];
} _GB2312State;

int
_GB2312_init(struct __xlocale_st_runelocale *xrl)
{

	xrl->__mbrtowc = _GB2312_mbrtowc;
	xrl->__wcrtomb = _GB2312_wcrtomb;
	xrl->__mbsinit = _GB2312_mbsinit;
	xrl->__mb_cur_max = GB2312_MB_CUR_MAX;
	xrl->__mb_sb_limit = 128;
	return (0);
}

static int
_GB2312_mbsinit(const mbstate_t *ps, locale_t loc __unused)
{

	return (ps == NULL || ((const _GB2312State *)ps)->count == 0);
}

static __inline int
_GB2312_check(const char *str, size_t n)
{
	const u_char *s = (const u_char *)str;

	if (n == 0)
		/* Incomplete multibyte sequence */
		return (-2);
	if (s[0] >= 0xa1 && s[0] <= 0xfe) {
		if (n < 2)
			/* Incomplete multibyte sequence */
			return (-2);
		if (s[1] < 0xa1 || s[1] > 0xfe)
			/* Invalid multibyte sequence */
			return (-1);
		return (2);
	} else if (s[0] & 0x80) {
		/* Invalid multibyte sequence */
		return (-1);
	} 
	return (1);
}

static size_t
_GB2312_mbrtowc(wchar_t * __restrict pwc, const char * __restrict s, size_t n,
    mbstate_t * __restrict ps, locale_t loc __unused)
{
	_GB2312State *gs;
	wchar_t wc;
	int i, len, ocount;
	size_t ncopy;

	gs = (_GB2312State *)ps;

	if (gs->count < 0 || gs->count > sizeof(gs->bytes)) {
		errno = EINVAL;
		return ((size_t)-1);
	}

	if (s == NULL) {
		s = "";
		n = 1;
		pwc = NULL;
	}

	ncopy = MIN(MIN(n, GB2312_MB_CUR_MAX), sizeof(gs->bytes) - gs->count);
	strncpy((char*)(gs->bytes + gs->count), s, ncopy);
	ocount = gs->count;
	gs->count += ncopy;
	s = (char *)gs->bytes;
	n = gs->count;

	if ((len = _GB2312_check(s, n)) < 0)
		return ((size_t)len);
	wc = 0;
	i = len;
	while (i-- > 0)
		wc = (wc << 8) | (unsigned char)*s++;
	if (pwc != NULL)
		*pwc = wc;
	gs->count = 0;
	return (wc == L'\0' ? 0 : len - ocount);
}

static size_t
_GB2312_wcrtomb(char * __restrict s, wchar_t wc, mbstate_t * __restrict ps, locale_t loc __unused)
{
	_GB2312State *gs;

	gs = (_GB2312State *)ps;

	if (gs->count != 0) {
		errno = EINVAL;
		return ((size_t)-1);
	}

	if (s == NULL)
		/* Reset to initial shift state (no-op) */
		return (1);
	if (wc & 0x8000) {
		*s++ = (wc >> 8) & 0xff;
		*s = wc & 0xff;
		return (2);
	}
	*s = wc & 0xff;
	return (1);
}
