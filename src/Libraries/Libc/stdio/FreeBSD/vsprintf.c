/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)vsprintf.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/vsprintf.c,v 1.16 2008/04/17 22:17:54 jhb Exp $");

#include "xlocale_private.h"

#include <stdio.h>
#include <limits.h>
#include "local.h"

int
vsprintf_l(char * __restrict str, locale_t loc, const char * __restrict fmt, __va_list ap)
{
	int ret;
	FILE f;
	struct __sFILEX ext;
	f._extra = &ext;
	INITEXTRA(&f);

	f._file = -1;
	f._flags = __SWR | __SSTR;
	f._bf._base = f._p = (unsigned char *)str;
	f._bf._size = f._w = INT_MAX;
	f._orientation = 0;
	memset(&f._mbstate, 0, sizeof(mbstate_t));
	ret = __v2printf(XPRINTF_PLAIN, NULL, &f, loc, fmt, ap);
	*f._p = 0;
	return (ret);
}

int
vsprintf(char * __restrict str, const char * __restrict fmt, __va_list ap)
{
	return vsprintf_l(str, __current_locale(), fmt, ap);
}
