/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/fgetwc.c,v 1.13 2008/04/17 22:17:53 jhb Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"
#include "mblocal.h"

/*
 * MT-safe version.
 */
wint_t
fgetwc(FILE *fp)
{
	wint_t r;

	FLOCKFILE(fp);
	ORIENT(fp, 1);
	r = __fgetwc(fp, __current_locale());
	FUNLOCKFILE(fp);

	return (r);
}

wint_t
fgetwc_l(FILE *fp, locale_t loc)
{
	wint_t r;

	NORMALIZE_LOCALE(loc);
	FLOCKFILE(fp);
	ORIENT(fp, 1);
	r = __fgetwc(fp, loc);
	FUNLOCKFILE(fp);

	return (r);
}

/*
 * Non-MT-safe version.
 */
wint_t
__fgetwc(FILE *fp, locale_t loc)
{
	wchar_t wc;
	size_t nconv;
	struct __xlocale_st_runelocale *xrl = loc->__lc_ctype;
	size_t (*__mbrtowc)(wchar_t * __restrict, const char * __restrict, size_t, mbstate_t * __restrict, locale_t) = xrl->__mbrtowc;

	if (fp->_r <= 0 && __srefill(fp))
		return (WEOF);
	if (xrl->__mb_cur_max == 1) {
		/* Fast path for single-byte encodings. */
		wc = *fp->_p++;
		fp->_r--;
		return (wc);
	}
	do {
		nconv = __mbrtowc(&wc, (char*)fp->_p, fp->_r, &fp->_mbstate, loc);
		if (nconv == (size_t)-1)
			break;
		else if (nconv == (size_t)-2)
			continue;
		else if (nconv == 0) {
			/*
			 * Assume that the only valid representation of
			 * the null wide character is a single null byte.
			 */
			fp->_p++;
			fp->_r--;
			return (L'\0');
		} else {
			fp->_p += nconv;
			fp->_r -= nconv;
			return (wc);
		}
	} while (__srefill(fp) == 0);
	fp->_flags |= __SERR;
	errno = EILSEQ;
	return (WEOF);
}
