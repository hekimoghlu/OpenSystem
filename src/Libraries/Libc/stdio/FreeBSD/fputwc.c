/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/fputwc.c,v 1.11 2008/04/17 22:17:53 jhb Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"
#include "mblocal.h"

/*
 * Non-MT-safe version.
 */
__private_extern__ wint_t
__fputwc(wchar_t wc, FILE *fp, locale_t loc)
{
	char buf[MB_LEN_MAX];
	size_t i, len;
	struct xlocale_ctype *xrl = XLOCALE_CTYPE(loc);

	if (xrl->__mb_cur_max == 1 && wc > 0 && wc <= UCHAR_MAX) {
		/*
		 * Assume single-byte locale with no special encoding.
		 * A more careful test would be to check
		 * _CurrentRuneLocale->encoding.
		 */
		*buf = (unsigned char)wc;
		len = 1;
	} else {
		if ((len = XLOCALE_CTYPE(loc)->__wcrtomb(buf, wc, &fp->_mbstate, loc)) == (size_t)-1) {
			fp->_flags |= __SERR;
			return (WEOF);
		}
	}

	for (i = 0; i < len; i++)
		if (__sputc((unsigned char)buf[i], fp) == EOF)
			return (WEOF);

	return ((wint_t)wc);
}

/*
 * MT-safe version.
 */
wint_t
fputwc(wchar_t wc, FILE *fp)
{
	wint_t r;

	FLOCKFILE(fp);
	ORIENT(fp, 1);
	r = __fputwc(wc, fp, __current_locale());
	FUNLOCKFILE(fp);

	return (r);
}

wint_t
fputwc_l(wchar_t wc, FILE *fp, locale_t loc)
{
	wint_t r;

	NORMALIZE_LOCALE(loc);
	FLOCKFILE(fp);
	ORIENT(fp, 1);
	r = __fputwc(wc, fp, loc);
	FUNLOCKFILE(fp);

	return (r);
}
