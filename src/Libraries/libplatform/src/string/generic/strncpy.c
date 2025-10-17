/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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

#if !_PLATFORM_OPTIMIZED_STRNCPY

char *
_platform_strncpy(char * restrict dst, const char * restrict src, size_t maxlen) {
    const size_t srclen = _platform_strnlen(src, maxlen);
    if (srclen < maxlen) {
        //  The stpncpy() and strncpy() functions copy at most maxlen
        //  characters from src into dst.
        _platform_memmove(dst, src, srclen);
        //  If src is less than maxlen characters long, the remainder
        //  of dst is filled with '\0' characters.
        _platform_memset(dst+srclen, 0, maxlen-srclen);
    } else {
        //  Otherwise, dst is not terminated.
        _platform_memmove(dst, src, maxlen);
    }
    //  The strcpy() and strncpy() functions return dst.
    return dst;
}

#if VARIANT_STATIC
char *
strncpy(char * restrict dst, const char * restrict src, size_t maxlen) {
	return _platform_strncpy(dst, src, maxlen);
}
#endif

#endif
