/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include <platform/string.h>

#if !_PLATFORM_OPTIMIZED_MEMCMP

int
_platform_memcmp(const void *s1, const void *s2, size_t n)
{
	if (n != 0) {
		const unsigned char *p1 = s1, *p2 = s2;

		do {
			if (*p1++ != *p2++)
				return (*--p1 - *--p2);
		} while (--n != 0);
	}
	return (0);
}

#if VARIANT_STATIC
int
memcmp(const void *s1, const void *s2, size_t n)
{
	return _platform_memcmp(s1, s2, n);
}

int
bcmp(const void *s1, const void *s2, size_t n)
{
	return _platform_memcmp(s1, s2, n);
}
#endif

#endif
