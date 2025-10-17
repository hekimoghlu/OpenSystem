/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/vdprintf.c,v 1.1 2009/03/04 03:38:51 das Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include "un-namespace.h"

#include "local.h"

__private_extern__ int
_vdprintf(printf_comp_t __restrict pc, printf_domain_t __restrict domain, int fd, locale_t __restrict loc, const char * __restrict fmt, va_list ap)
{
	FILE f;
	unsigned char buf[BUFSIZ];
	int ret;
	struct __sFILEX ext;
	f._extra = &ext;
	INITEXTRA(&f);

	if (fd > SHRT_MAX) {
		errno = EMFILE;
		return (EOF);
	}

	f._p = buf;
	f._w = sizeof(buf);
	f._flags = __SWR;
	f._file = fd;
	f._cookie = &f;
	f._write = __swrite;
	f._bf._base = buf;
	f._bf._size = sizeof(buf);
	f._orientation = 0;
	bzero(&f._mbstate, sizeof(f._mbstate));

	if ((ret = __v2printf(pc, domain, &f, loc, fmt, ap)) < 0)
		return (ret);

	return (__fflush(&f) ? EOF : ret);
}

int
vdprintf_l(int fd, locale_t __restrict loc, const char * __restrict fmt, va_list ap)
{
	return _vdprintf(XPRINTF_PLAIN, NULL, fd, loc, fmt, ap);
}

int
vdprintf(int fd, const char * __restrict fmt, va_list ap) {
	return _vdprintf(XPRINTF_PLAIN, NULL, fd, __current_locale(), fmt, ap);
}
