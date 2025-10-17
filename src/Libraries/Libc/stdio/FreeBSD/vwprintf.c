/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/vwprintf.c,v 1.1 2002/09/21 13:00:30 tjr Exp $");

#include "xlocale_private.h"

#include <stdarg.h>
#include <stdio.h>
#include <wchar.h>

int
vwprintf(const wchar_t * __restrict fmt, va_list ap)
{

	return (vfwprintf_l(stdout, __current_locale(), fmt, ap));
}

int
vwprintf_l(locale_t loc, const wchar_t * __restrict fmt, va_list ap)
{

	/* no need to call NORMALIZE_LOCALE(loc) because vfwprintf_l will */
	return (vfwprintf_l(stdout, loc, fmt, ap));
}
