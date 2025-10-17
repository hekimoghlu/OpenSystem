/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#if !VARIANT_STATIC
// to satisfy compiler-generated memset inside libplatform (e.g. makecontext)
__attribute__((visibility("hidden")))
void *
memset(void *b, int c, size_t len)
{
	return _platform_memset(b, c, len);
}
#endif

#if !_PLATFORM_OPTIMIZED_MEMSET

void *
_platform_memset(void *b, int c, size_t len) {
	unsigned char pattern[4];

	pattern[0] = (unsigned char)c;
	pattern[1] = (unsigned char)c;
	pattern[2] = (unsigned char)c;
	pattern[3] = (unsigned char)c;

	_platform_memset_pattern4(b, pattern, len);
	return b;
}

#if VARIANT_STATIC
void *
memset(void *b, int c, size_t len) {
	return _platform_memset(b, c, len);
}
#endif

#endif


#if !_PLATFORM_OPTIMIZED_BZERO

void
_platform_bzero(void *s, size_t n)
{
	_platform_memset(s, 0, n);
}

#if VARIANT_STATIC
void
bzero(void *s, size_t n) {
	_platform_bzero(s, n);
}

void
__bzero(void *s, size_t n) {
	_platform_bzero(s, n);
}
#endif

#endif
