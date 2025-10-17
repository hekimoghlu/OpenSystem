/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)vsscanf.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
__FBSDID("FreeBSD: src/lib/libc/stdio/vsscanf.c,v 1.11 2002/08/21 16:19:57 mike Exp ");
#endif
__FBSDID("$FreeBSD: src/lib/libc/stdio/vswscanf.c,v 1.6 2009/01/15 18:53:52 rdivacky Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "local.h"

static int	eofread(void *, char *, int);

static int
eofread(void *cookie, char *buf, int len)
{

	return (0);
}

int
vswscanf_l(const wchar_t * __restrict str, locale_t loc, const wchar_t * __restrict fmt,
    va_list ap)
{
	static const mbstate_t initial;
	mbstate_t mbs;
	FILE f;
	char *mbstr;
	size_t mlen;
	int r;
	const wchar_t *strp;
	struct __sFILEX ext;
	f._extra = &ext;
	INITEXTRA(&f);

	NORMALIZE_LOCALE(loc);
	/*
	 * XXX Convert the wide character string to multibyte, which
	 * __vfwscanf() will convert back to wide characters.
	 */
	if ((mbstr = malloc(wcslen(str) * MB_CUR_MAX_L(loc) + 1)) == NULL)
		return (EOF);
	mbs = initial;
	strp = str;
	if ((mlen = wcsrtombs_l(mbstr, &strp, SIZE_T_MAX, &mbs, loc)) == (size_t)-1) {
		free(mbstr);
		return (EOF);
	}
	f._file = -1;
	f._flags = __SRD;
	f._bf._base = f._p = (unsigned char *)mbstr;
	f._bf._size = f._r = mlen;
	f._read = eofread;
	f._ub._base = NULL;
	f._lb._base = NULL;
	f._orientation = 0;
	memset(&f._mbstate, 0, sizeof(mbstate_t));
	r = __vfwscanf(&f, loc, fmt, ap);
	free(mbstr);

	return (r);
}

int
vswscanf(const wchar_t * __restrict str, const wchar_t * __restrict fmt,
    va_list ap)
{
	return vswscanf_l(str, __current_locale(), fmt, ap);
}

