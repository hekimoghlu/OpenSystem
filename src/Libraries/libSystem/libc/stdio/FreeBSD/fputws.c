/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/fputws.c,v 1.8 2009/01/15 18:53:52 rdivacky Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <wchar.h>
#include "un-namespace.h"
#include "fvwrite.h"
#include "libc_private.h"
#include "local.h"
#include "mblocal.h"

int
fputws_l(const wchar_t * __restrict ws, FILE * __restrict fp, locale_t loc)
{
	size_t nbytes;
	char buf[BUFSIZ];
	struct __suio uio;
	struct __siov iov;
	const wchar_t *wsp = ws;
	size_t (*__wcsnrtombs)(char * __restrict, const wchar_t ** __restrict,
	    size_t, size_t, mbstate_t * __restrict, locale_t);

	NORMALIZE_LOCALE(loc);
	__wcsnrtombs = loc->__lc_ctype->__wcsnrtombs;
	FLOCKFILE(fp);
	ORIENT(fp, 1);
	if (prepwrite(fp) != 0)
		goto error;
	uio.uio_iov = &iov;
	uio.uio_iovcnt = 1;
	iov.iov_base = buf;
	do {
		nbytes = __wcsnrtombs(buf, &wsp, SIZE_T_MAX, sizeof(buf),
		    &fp->_mbstate, loc);
		if (nbytes == (size_t)-1)
			goto error;
		iov.iov_len = uio.uio_resid = nbytes;
		if (__sfvwrite(fp, &uio) != 0)
			goto error;
	} while (wsp != NULL);
	FUNLOCKFILE(fp);
	return (0);

error:
	FUNLOCKFILE(fp);
	return (-1);
}

int
fputws(const wchar_t * __restrict ws, FILE * __restrict fp)
{
	return fputws_l(ws, fp, __current_locale());
}
