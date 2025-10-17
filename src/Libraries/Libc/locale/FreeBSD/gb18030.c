/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
/*
 * PRC National Standard GB 18030-2000 encoding of Chinese text.
 *
 * See gb18030(5) for details.
 */

#include <sys/param.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/gb18030.c,v 1.8 2007/10/13 16:28:21 ache Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <runetype.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "mblocal.h"

#define GB18030_MB_CUR_MAX	4

static size_t	_GB18030_mbrtowc(wchar_t * __restrict, const char * __restrict,
		    size_t, mbstate_t * __restrict, locale_t);
static int	_GB18030_mbsinit(const mbstate_t *, locale_t);
static size_t	_GB18030_wcrtomb(char * __restrict, wchar_t,
		    mbstate_t * __restrict, locale_t);

typedef struct {
	int	count;
	u_char	bytes[4];
} _GB18030State;

int
_GB18030_init(struct xlocale_ctype *xrl)
{

	xrl->__mbrtowc = _GB18030_mbrtowc;
	xrl->__wcrtomb = _GB18030_wcrtomb;
	xrl->__mbsinit = _GB18030_mbsinit;
	xrl->__mb_cur_max = GB18030_MB_CUR_MAX;
	xrl->__mb_sb_limit = 128;

	return (0);
}

static int
_GB18030_mbsinit(const mbstate_t *ps, locale_t loc __unused)
{

	return (ps == NULL || ((const _GB18030State *)ps)->count == 0);
}

static size_t
_GB18030_mbrtowc(wchar_t * __restrict pwc, const char * __restrict s,
    size_t n, mbstate_t * __restrict ps, locale_t loc __unused)
{
	_GB18030State *gs;
	wchar_t wch;
	int ch, len, ocount;
	size_t ncopy;

	gs = (_GB18030State *)ps;

	if (gs->count < 0 || gs->count > sizeof(gs->bytes)) {
		errno = EINVAL;
		return ((size_t)-1);
	}

	if (s == NULL) {
		s = "";
		n = 1;
		pwc = NULL;
	}

	ncopy = MIN(MIN(n, GB18030_MB_CUR_MAX), sizeof(gs->bytes) - gs->count);
	strncpy((char*)(gs->bytes + gs->count), s, ncopy);
	ocount = gs->count;
	gs->count += ncopy;
	s = (char *)gs->bytes;
	n = gs->count;

	if (n == 0)
		/* Incomplete multibyte sequence */
		return ((size_t)-2);

	/*
	 * Single byte:		[00-7f]
	 * Two byte:		[81-fe][40-7e,80-fe]
	 * Four byte:		[81-fe][30-39][81-fe][30-39]
	 */
	ch = (unsigned char)*s++;
	if (ch <= 0x7f) {
		len = 1;
		wch = ch;
	} else if (ch >= 0x81 && ch <= 0xfe) {
		wch = ch;
		if (n < 2)
			return ((size_t)-2);
		ch = (unsigned char)*s++;
		if ((ch >= 0x40 && ch <= 0x7e) || (ch >= 0x80 && ch <= 0xfe)) {
			wch = (wch << 8) | ch;
			len = 2;
		} else if (ch >= 0x30 && ch <= 0x39) {
			/*
			 * Strip high bit off the wide character we will
			 * eventually output so that it is positive when
			 * cast to wint_t on 32-bit twos-complement machines.
			 */
			wch = ((wch & 0x7f) << 8) | ch;
			if (n < 3)
				return ((size_t)-2);
			ch = (unsigned char)*s++;
			if (ch < 0x81 || ch > 0xfe)
				goto ilseq;
			wch = (wch << 8) | ch;
			if (n < 4)
				return ((size_t)-2);
			ch = (unsigned char)*s++;
			if (ch < 0x30 || ch > 0x39)
				goto ilseq;
			wch = (wch << 8) | ch;
			len = 4;
		} else
			goto ilseq;
	} else
		goto ilseq;

	if (pwc != NULL)
		*pwc = wch;
	gs->count = 0;
	return (wch == L'\0' ? 0 : len - ocount);
ilseq:
	errno = EILSEQ;
	return ((size_t)-1);
}

static size_t
_GB18030_wcrtomb(char * __restrict s, wchar_t wc, mbstate_t * __restrict ps, locale_t loc __unused)
{
	_GB18030State *gs;
	size_t len;
	int c;

	gs = (_GB18030State *)ps;

	if (gs->count != 0) {
		errno = EINVAL;
		return ((size_t)-1);
	}

	if (s == NULL)
		/* Reset to initial shift state (no-op) */
		return (1);
	if ((wc & ~0x7fffffff) != 0)
		goto ilseq;
	if (wc & 0x7f000000) {
		/* Replace high bit that mbrtowc() removed. */
		wc |= 0x80000000;
		c = (wc >> 24) & 0xff;
		if (c < 0x81 || c > 0xfe)
			goto ilseq;
		*s++ = c;
		c = (wc >> 16) & 0xff;
		if (c < 0x30 || c > 0x39)
			goto ilseq;
		*s++ = c;
		c = (wc >> 8) & 0xff;
		if (c < 0x81 || c > 0xfe)
			goto ilseq;
		*s++ = c;
		c = wc & 0xff;
		if (c < 0x30 || c > 0x39)
			goto ilseq;
		*s++ = c;
		len = 4;
	} else if (wc & 0x00ff0000)
		goto ilseq;
	else if (wc & 0x0000ff00) {
		c = (wc >> 8) & 0xff;
		if (c < 0x81 || c > 0xfe)
			goto ilseq;
		*s++ = c;
		c = wc & 0xff;
		if (c < 0x40 || c == 0x7f || c == 0xff)
			goto ilseq;
		*s++ = c;
		len = 2;
	} else if (wc <= 0x7f) {
		*s++ = wc;
		len = 1;
	} else
		goto ilseq;

	return (len);
ilseq:
	errno = EILSEQ;
	return ((size_t)-1);
}
