/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

#if !_PLATFORM_OPTIMIZED_MEMCCPY

#include <stdlib.h>

void *
_platform_memccpy(void *t, const void *f, int c, size_t n)
{
	void *last;

	if (n == 0) {
		return NULL;
	}

	last = _platform_memchr(f, c, n);

	if (last == NULL) {
		_platform_memmove(t, f, n);
		return NULL;
	} else {
		n = (char *)last - (char *)f + 1;
		_platform_memmove(t, f, n);
		return (void *)((char *)t + n);
	}
}

#if VARIANT_STATIC
void *
memccpy(void *t, const void *f, int c, size_t n)
{
	return _platform_memccpy(t, f, c, n);
}
#endif

#endif
