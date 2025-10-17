/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#if !_PLATFORM_OPTIMIZED_MEMSET_PATTERN4

void
_platform_memset_pattern4(void *b, const void *pattern4, size_t len)
{
	char * start = (char *)b;
	char * p = (char *)b;
	while ((start + len) - p >= 4) {
		_platform_memmove(p, pattern4, 4);
		p += 4;
	}
	if ((start + len) - p != 0) {
		_platform_memmove(p, pattern4, (start + len) - p);
	}
}

#if VARIANT_STATIC
void
memset_pattern4(void *b, const void *pattern4, size_t len)
{
	return _platform_memset_pattern4(b, pattern4, len);
}
#endif

#endif


#if !_PLATFORM_OPTIMIZED_MEMSET_PATTERN8

void
_platform_memset_pattern8(void *b, const void *pattern8, size_t len)
{
	char * start = (char *)b;
	char * p = (char *)b;
	while ((start + len) - p >= 8) {
		_platform_memmove(p, pattern8, 8);
		p += 8;
	}
	if ((start + len) - p != 0) {
		_platform_memmove(p, pattern8, (start + len) - p);
	}
}

#if VARIANT_STATIC
void
memset_pattern8(void *b, const void *pattern8, size_t len)
{
	return _platform_memset_pattern8(b, pattern8, len);
}
#endif

#endif


#if !_PLATFORM_OPTIMIZED_MEMSET_PATTERN16

void
_platform_memset_pattern16(void *b, const void *pattern16, size_t len)
{
	char * start = (char *)b;
	char * p = (char *)b;
	while ((start + len) - p >= 16) {
		_platform_memmove(p, pattern16, 16);
		p += 16;
	}
	if ((start + len) - p != 0) {
		_platform_memmove(p, pattern16, (start + len) - p);
	}
}

#if VARIANT_STATIC
void
memset_pattern16(void *b, const void *pattern16, size_t len)
{
	return _platform_memset_pattern16(b, pattern16, len);
}
#endif

#endif
