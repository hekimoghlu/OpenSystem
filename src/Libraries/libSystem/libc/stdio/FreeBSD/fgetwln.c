/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/fgetwln.c,v 1.2 2004/08/06 17:00:09 tjr Exp $");

#include "namespace.h"
#include <stdio.h>
#include <wchar.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

wchar_t *
fgetwln_l(FILE * __restrict fp, size_t *lenp, locale_t loc)
{
	wint_t wc;
	size_t len;

	FLOCKFILE(fp);
	ORIENT(fp, 1);

	len = 0;
	while ((wc = __fgetwc(fp, loc)) != WEOF) {
#define	GROW	512
		if (len * sizeof(wchar_t) >= fp->_lb._size &&
		    __slbexpand(fp, (len + GROW) * sizeof(wchar_t)))
			goto error;
		*((wchar_t *)fp->_lb._base + len++) = wc;
		if (wc == L'\n')
			break;
	}
	if (len == 0)
		goto error;

	FUNLOCKFILE(fp);
	*lenp = len;
	return ((wchar_t *)fp->_lb._base);

error:
	FUNLOCKFILE(fp);
	*lenp = 0;
	return (NULL);
}

wchar_t *
fgetwln(FILE * __restrict fp, size_t *lenp)
{
	return fgetwln_l(fp, lenp, __current_locale());
}
