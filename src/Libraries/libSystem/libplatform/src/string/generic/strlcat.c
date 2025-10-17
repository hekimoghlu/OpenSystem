/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#if !_PLATFORM_OPTIMIZED_STRLCAT

size_t
_platform_strlcat(char * restrict dst, const char * restrict src, size_t maxlen) {
    const size_t srclen = _platform_strlen(src);
    const size_t dstlen = _platform_strnlen(dst, maxlen);
    if (dstlen == maxlen) return maxlen+srclen;
    if (srclen < maxlen-dstlen) {
        _platform_memmove(dst+dstlen, src, srclen+1);
    } else {
        _platform_memmove(dst+dstlen, src, maxlen-dstlen-1);
        dst[maxlen-1] = '\0';
    }
    return dstlen + srclen;
}

#if VARIANT_STATIC
size_t
strlcat(char * restrict dst, const char * restrict src, size_t maxlen) {
	return _platform_strlcat(dst, src, maxlen);
}
#endif

#endif
