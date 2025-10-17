/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
/*
 * Machine-dependent glue to integrate David Gay's gdtoa
 * package into libc for architectures where a long double
 * is an IEEE extended precision number.
 */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gdtoa/machdep_ldisx.c,v 1.2 2003/04/09 05:58:43 das Exp $");

#include "xlocale_private.h"

#include "gdtoaimp.h"

long double
strtold(const char * __restrict s, char ** __restrict sp)
{
	long double result;

	strtopx(s, sp, &result, __current_locale());
	return result;
}

long double
strtold_l(const char * __restrict s, char ** __restrict sp, locale_t loc)
{
	long double result;

	NORMALIZE_LOCALE(loc);
	strtopx(s, sp, &result, loc);
	return result;
}
