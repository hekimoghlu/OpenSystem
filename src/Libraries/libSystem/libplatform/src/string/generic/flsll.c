/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/string/flsll.c,v 1.1 2008/11/03 10:22:19 kib Exp $");

#include <strings.h>

/*
 * Find Last Set bit
 */
int
flsll(long long mask)
{
#if __has_builtin(__builtin_flsll)
	return __builtin_flsll(mask);
#elif __has_builtin(__builtin_clzll)
	if (mask == 0)
		return (0);

	return (sizeof(mask) << 3) - __builtin_clzll(mask);
#else
	int bit;

	if (mask == 0)
		return (0);
	for (bit = 1; mask != 1; bit++)
		mask = (unsigned long long)mask >> 1;
	return (bit);
#endif
}

#if VARIANT_DYLD && TARGET_OS_SIMULATOR
int
flsl(long mask)
{
#if __has_builtin(__builtin_flsl)
	return __builtin_flsl(mask);
#elif __has_builtin(__builtin_clzl)
	if (mask == 0)
		return (0);

	return (sizeof(mask) << 3) - __builtin_clzl(mask);
#else
	int bit;

	if (mask == 0)
		return (0);
	for (bit = 1; mask != 1; bit++)
		mask = (unsigned long)mask >> 1;
	return (bit);
#endif
}

int
fls(int mask)
{
#if __has_builtin(__builtin_fls)
	return __builtin_fls(mask);
#elif __has_builtin(__builtin_clz)
	if (mask == 0)
		return (0);

	return (sizeof(mask) << 3) - __builtin_clz(mask);
#else
	int bit;

	if (mask == 0)
		return (0);
	for (bit = 1; mask != 1; bit++)
		mask = (unsigned)mask >> 1;
	return (bit);
#endif
}
#endif
