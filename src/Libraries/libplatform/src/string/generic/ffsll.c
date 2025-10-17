/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
#include <TargetConditionals.h>
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/string/ffsll.c,v 1.1 2008/11/03 10:22:19 kib Exp $");

#include <strings.h>

/*
 * Find First Set bit
 */
int
ffsll(long long mask)
{
#if __has_builtin(__builtin_ffsll)
	return __builtin_ffsll(mask);
#elif __has_builtin(__builtin_ctzll)
	if (mask == 0)
		return (0);

	return __builtin_ctzll(mask) + 1;
#else
	int bit;

	if (mask == 0)
		return (0);
	for (bit = 1; !(mask & 1); bit++)
		mask = (unsigned long long)mask >> 1;
	return (bit);
#endif
}

#if VARIANT_DYLD && TARGET_OS_SIMULATOR
int
ffsl(long mask)
{
#if __has_builtin(__builtin_ffsl)
	return __builtin_ffsl(mask);
#elif __has_builtin(__builtin_ctzl)
	if (mask == 0)
		return (0);

	return __builtin_ctzl(mask) + 1;
#else
	int bit;

	if (mask == 0)
		return (0);
	for (bit = 1; !(mask & 1); bit++)
		mask = (unsigned long)mask >> 1;
	return (bit);
#endif
}

int
ffs(int mask)
{
#if __has_builtin(__builtin_ffs)
	return __builtin_ffs(mask);
#elif __has_builtin(__builtin_ctz)
	if (mask == 0)
		return (0);

	return __builtin_ctz(mask) + 1;
#else
	int bit;

	if (mask == 0)
		return (0);
	for (bit = 1; !(mask & 1); bit++)
		mask = (unsigned)mask >> 1;
	return (bit);
#endif
}
#endif

