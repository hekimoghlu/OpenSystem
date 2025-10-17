/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wcsftime.c,v 1.6 2009/01/15 20:45:59 rdivacky Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <wchar.h>

/*
 * Convert date and time to a wide-character string.
 *
 * This is the wide-character counterpart of strftime(). So that we do not
 * have to duplicate the code of strftime(), we convert the format string to
 * multibyte, call strftime(), then convert the result back into wide
 * characters.
 *
 * This technique loses in the presence of stateful multibyte encoding if any
 * of the conversions in the format string change conversion state. When
 * stateful encoding is implemented, we will need to reset the state between
 * format specifications in the format string.
 */
size_t
wcsftime_l(wchar_t * __restrict wcs, size_t maxsize,
    const wchar_t * __restrict format, const struct tm * __restrict timeptr,
    locale_t loc)
{
	static const mbstate_t initial;
	mbstate_t mbs;
	char *dst, *sformat;
	const char *dstp;
	const wchar_t *formatp;
	size_t n, sflen;
	int sverrno;

	NORMALIZE_LOCALE(loc);
	sformat = dst = NULL;

	/*
	 * Convert the supplied format string to a multibyte representation
	 * for strftime(), which only handles single-byte characters.
	 */
	mbs = initial;
	formatp = format;
	sflen = wcsrtombs_l(NULL, &formatp, 0, &mbs, loc);
	if (sflen == (size_t)-1)
		goto error;
	if ((sformat = malloc(sflen + 1)) == NULL)
		goto error;
	mbs = initial;
	wcsrtombs_l(sformat, &formatp, sflen + 1, &mbs, loc);

	/*
	 * Allocate memory for longest multibyte sequence that will fit
	 * into the caller's buffer and call strftime() to fill it.
	 * Then, copy and convert the result back into wide characters in
	 * the caller's buffer.
	 */
	if (SIZE_T_MAX / MB_CUR_MAX_L(loc) <= maxsize) {
		/* maxsize is prepostorously large - avoid int. overflow. */
		errno = EINVAL;
		goto error;
	}
	if ((dst = malloc(maxsize * MB_CUR_MAX_L(loc))) == NULL)
		goto error;
	if (strftime_l(dst, maxsize, sformat, timeptr, loc) == 0)
		goto error;
	dstp = dst;
	mbs = initial;
	n = mbsrtowcs_l(wcs, &dstp, maxsize, &mbs, loc);
	if (n == (size_t)-2 || n == (size_t)-1 || dstp != NULL)
		goto error;

	free(sformat);
	free(dst);
	return (n);

error:
	sverrno = errno;
	free(sformat);
	free(dst);
	errno = sverrno;
	return (0);
}

size_t
wcsftime(wchar_t * __restrict wcs, size_t maxsize,
    const wchar_t * __restrict format, const struct tm * __restrict timeptr)
{
	return wcsftime_l(wcs, maxsize, format, timeptr, __current_locale());
}
