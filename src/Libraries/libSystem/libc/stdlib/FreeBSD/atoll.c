/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
__FBSDID("$FreeBSD: src/lib/libc/stdlib/atoll.c,v 1.5 2007/01/09 00:28:09 imp Exp $");

#include "xlocale_private.h"

#include <stdlib.h>

long long
atoll(str)
	const char *str;
{
	return strtoll_l(str, (char **)NULL, 10, __current_locale());
}

long long
atoll_l(str, loc)
	const char *str;
	locale_t loc;
{
	/* no need to call NORMALIZE_LOCALE(loc) because strtoll_l will */
	return strtoll_l(str, (char **)NULL, 10, loc);
}
