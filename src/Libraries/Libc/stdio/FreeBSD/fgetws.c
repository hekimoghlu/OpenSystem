/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/fgetws.c,v 1.8 2009/11/25 04:45:45 wollman Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"
#include "mblocal.h"

wchar_t *
fgetws_l(wchar_t * __restrict ws, int n, FILE * __restrict fp, locale_t loc)
{
	wchar_t *wsp;
	size_t nconv;
	const char *src;
	unsigned char *nl;
	struct xlocale_ctype *rl;
	size_t (*__mbsnrtowcs)(wchar_t * __restrict, const char ** __restrict, size_t, size_t, __darwin_mbstate_t * __restrict, locale_t);

	NORMALIZE_LOCALE(loc);
	rl = XLOCALE_CTYPE(loc);
	__mbsnrtowcs = rl->__mbsnrtowcs;
	FLOCKFILE(fp);
	ORIENT(fp, 1);

	if (n <= 0) {
		errno = EINVAL;
		goto error;
	}

	if (fp->_r <= 0 && __srefill(fp))
		/* EOF */
		goto error;
	wsp = ws;
	do {
		src = (const char *)fp->_p;
		nl = memchr(fp->_p, '\n', fp->_r);
		nconv = __mbsnrtowcs(wsp, &src,
		    nl != NULL ? (nl - fp->_p + 1) : fp->_r,
		    n - 1, &fp->_mbstate, loc);
		if (nconv == (size_t)-1)
			/* Conversion error */
			goto error;
		if (src == NULL) {
			/*
			 * We hit a null byte. Increment the character count,
			 * since mbsnrtowcs()'s return value doesn't include
			 * the terminating null, then resume conversion
			 * after the null.
			 */
			nconv++;
			src = memchr(fp->_p, '\0', fp->_r);
			src++;
		}
		fp->_r -= (unsigned char *)src - fp->_p;
		fp->_p = (unsigned char *)src;
		n -= nconv;
		wsp += nconv;
	} while (wsp[-1] != L'\n' && n > 1 && (fp->_r > 0 ||
	    __srefill(fp) == 0));
	if (wsp == ws)
		/* EOF */
		goto error;
	if (!rl->__mbsinit(&fp->_mbstate, loc))
		/* Incomplete character */
		goto error;
	*wsp = L'\0';
	FUNLOCKFILE(fp);

	return (ws);

error:
	FUNLOCKFILE(fp);
	return (NULL);
}

wchar_t *
fgetws(wchar_t * __restrict ws, int n, FILE * __restrict fp)
{
	return fgetws_l(ws, n, fp, __current_locale());
}
