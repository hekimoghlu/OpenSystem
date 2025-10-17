/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/asprintf.c,v 1.15 2009/03/02 04:11:42 das Exp $");

#include "xlocale_private.h"

#include <stdio.h>
#include <stdarg.h>

int
asprintf(char ** __restrict s, char const * __restrict fmt, ...)
{
	int ret;
	va_list ap;

	va_start(ap, fmt);
	ret = vasprintf_l(s, __current_locale(), fmt, ap);
	va_end(ap);
	return (ret);
}

int
asprintf_l(char ** __restrict s, locale_t loc, char const * __restrict fmt, ...)
{
	int ret;
	va_list ap;

	va_start(ap, fmt);
	ret = vasprintf_l(s, loc, fmt, ap);
	va_end(ap);
	return (ret);
}
