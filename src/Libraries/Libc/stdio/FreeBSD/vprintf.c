/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
static char sccsid[] = "@(#)vprintf.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/vprintf.c,v 1.11 2007/01/09 00:28:08 imp Exp $");

#include "xlocale_private.h"

#include <stdio.h>

int
vprintf(const char * __restrict fmt, __va_list ap)
{

	return (vfprintf_l(stdout, __current_locale(), fmt, ap));
}

int
vprintf_l(locale_t loc, const char * __restrict fmt, __va_list ap)
{

	/* no need to call NORMALIZE_LOCALE(loc) because vfprintf_l will */
	return (vfprintf_l(stdout, loc, fmt, ap));
}
