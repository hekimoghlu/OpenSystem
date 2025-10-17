/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
strncat(char *restrict dst, const char *restrict src, size_t maxlen) {
	const size_t dstlen = strlen(dst);
	const size_t srclen = strnlen(src, maxlen);
    //  The strcat() and strncat() functions append a copy of the null-
    //  terminated string src to the end of the null-terminated string dst,
    //  then add a terminating '\0'.  The string dst must have sufficient
    //  space to hold the result.
    //
    //  The strncat() function appends not more than maxlen characters
    //  from src, and then adds a terminating '\0'.
    const size_t cpylen = srclen < maxlen ? srclen : maxlen;
    memcpy(dst+dstlen, src, cpylen);
    dst[dstlen+cpylen] = '\0';
    //  The strcat() and strncat() functions return dst.
    return dst;
}
