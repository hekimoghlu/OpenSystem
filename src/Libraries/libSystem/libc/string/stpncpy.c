/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#include <string.h>

char *
stpncpy(char * restrict dst, const char * restrict src, size_t maxlen) {
    const size_t srclen = strnlen(src, maxlen);
    if (srclen < maxlen) {
        //  The stpncpy() and strncpy() functions copy at most maxlen
        //  characters from src into dst.
        memcpy(dst, src, srclen);
        //  If src is less than maxlen characters long, the remainder
        //  of dst is filled with '\0' characters.
        memset(dst+srclen, 0, maxlen-srclen);
        //  The stpcpy() and stpncpy() functions return a pointer to the
        //  terminating '\0' character of dst.
        return dst+srclen;
    } else {
        //  The stpncpy() and strncpy() functions copy at most maxlen
        //  characters from src into dst.
        memcpy(dst, src, maxlen);
        //  If stpncpy() does not terminate dst with a NUL character, it
        //  instead returns a pointer to src[maxlen] (which does not
        //  necessarily refer to a valid memory location.)
        return dst+maxlen;
    }
}
