/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
#include "strings.h"

__attribute__((visibility("hidden")))
size_t
_libkernel_strlcpy(char * restrict dst, const char * restrict src, size_t maxlen)
{
	const size_t srclen = _libkernel_strlen(src);
	if (srclen < maxlen) {
		_libkernel_memmove(dst, src, srclen + 1);
	} else if (maxlen != 0) {
		_libkernel_memmove(dst, src, maxlen - 1);
		dst[maxlen - 1] = '\0';
	}
	return srclen;
}
