/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
static char sccsid[] = "@(#)fscanf.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fscanf.c,v 1.13 2007/01/09 00:28:06 imp Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <stdio.h>
#include <stdarg.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"
#include "libc_hooks_impl.h"

int
fscanf(FILE * __restrict fp, char const * __restrict fmt, ...)
{
	int ret;
	va_list ap;

	libc_hooks_will_write(fp, sizeof(*fp));

	va_start(ap, fmt);
	FLOCKFILE(fp);
	ret = __svfscanf_l(fp, __current_locale(), fmt, ap);
	va_end(ap);
	FUNLOCKFILE(fp);
	return (ret);
}

int
fscanf_l(FILE * __restrict fp, locale_t loc, char const * __restrict fmt, ...)
{
	int ret;
	va_list ap;

	libc_hooks_will_write(fp, sizeof(*fp));

	NORMALIZE_LOCALE(loc);
	va_start(ap, fmt);
	FLOCKFILE(fp);
	ret = __svfscanf_l(fp, loc, fmt, ap);
	va_end(ap);
	FUNLOCKFILE(fp);
	return (ret);
}
