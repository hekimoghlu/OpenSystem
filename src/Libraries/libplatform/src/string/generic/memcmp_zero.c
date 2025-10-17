/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

#if !_PLATFORM_OPTIMIZED_MEMCMP_ZERO_ALIGNED8

#if defined(__LP64__)

unsigned long
_platform_memcmp_zero_aligned8(const void *s, size_t n)
{
	size_t size = n;

	if (size == 0) {
		return 0;
	}

	const uint64_t *p = (const uint64_t *)s;
	uint64_t a = p[0];

	_Static_assert(sizeof(unsigned long) == sizeof(uint64_t), "");

	if (size < 4 * sizeof(uint64_t)) {
		if (size > 1 * sizeof(uint64_t)) {
			a |= p[1];
			if (size > 2 * sizeof(uint64_t)) {
				a |= p[2];
			}
		}
	} else {
		size_t count = size / sizeof(uint64_t);
		uint64_t b = p[1];
		uint64_t c = p[2];
		uint64_t d = p[3];

		/*
		 * note: for sizes not a multiple of 32 bytes, this will load
		 * the bytes [size % 32 .. 32) twice which is ok
		 */
		while (count > 4) {
			count -= 4;
			a |= p[count + 0];
			b |= p[count + 1];
			c |= p[count + 2];
			d |= p[count + 3];
		}

		a |= b | c | d;
	}

	return (a != 0);
}

#else // defined(__LP64__)

unsigned long
_platform_memcmp_zero_aligned8(const void *s, size_t n)
{
	uintptr_t p = (uintptr_t)s;
	uintptr_t end = (uintptr_t)s + n;
	uint32_t a, b;

	_Static_assert(sizeof(unsigned long) == sizeof(uint32_t), "");

	a = 0;
	b = 0;

	for (; p < end; p += sizeof(uint64_t)) {
		uint64_t v = *(const uint64_t *)p;
		a |= (uint32_t)v;
		b |= (uint32_t)(v >> 32);
	}

	return (a | b) != 0;
}

#endif // defined(__LP64__)

#endif // !_PLATFORM_OPTIMIZED_MEMCMP_ZERO_ALIGNED8
