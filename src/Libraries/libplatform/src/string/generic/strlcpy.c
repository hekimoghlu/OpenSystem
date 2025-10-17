/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#if !_PLATFORM_OPTIMIZED_STRLCPY

size_t
_platform_strlcpy(char * restrict dst, const char * restrict src, size_t maxlen) {
    const size_t srclen = _platform_strlen(src);
    if (srclen < maxlen) {
        _platform_memmove(dst, src, srclen+1);
    } else if (maxlen != 0) {
        _platform_memmove(dst, src, maxlen-1);
        dst[maxlen-1] = '\0';
    }
    return srclen;
}

#if VARIANT_STATIC
size_t
strlcpy(char * restrict dst, const char * restrict src, size_t maxlen) {
	return _platform_strlcpy(dst, src, maxlen);
}
#endif

#endif
