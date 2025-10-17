/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include "compiler.h"
#include "nasmlib.h"
#include "alloc.h"

/*
 * nasm_[v]asprintf() are variants of the semi-standard [v]asprintf()
 * functions, except that we return the pointer instead of a count.
 * The length of the string (with or without the final NUL) is available
 * by calling nasm_last_string_{len,size}() afterwards.
 *
 * nasm_[v]axprintf() are similar, but allocates a user-defined amount
 * of storage before the string, and returns a pointer to the
 * allocated buffer. The size of that area is not included in the value
 * returned by nasm_last_string_size().
 */

void *nasm_vaxprintf(size_t extra, const char *fmt, va_list ap)
{
    char *strp;
    va_list xap;
    size_t bytes;
    int len;

    va_copy(xap, ap);
    len = vsnprintf(NULL, 0, fmt, xap);
    nasm_assert(len >= 0);
    bytes = (size_t)len + 1;
    _nasm_last_string_size = bytes;
    va_end(xap);

    strp = nasm_malloc(extra+bytes);
    memset(strp, 0, extra);
    len = vsnprintf(strp+extra, bytes, fmt, ap);
    nasm_assert(bytes == (size_t)len + 1);
    return strp;
}

char *nasm_vasprintf(const char *fmt, va_list ap)
{
    return nasm_vaxprintf(0, fmt, ap);
}

void *nasm_axprintf(size_t extra, const char *fmt, ...)
{
    va_list ap;
    void *strp;

    va_start(ap, fmt);
    strp = nasm_vaxprintf(extra, fmt, ap);
    va_end(ap);

    return strp;
}

char *nasm_asprintf(const char *fmt, ...)
{
    va_list ap;
    char *strp;

    va_start(ap, fmt);
    strp = nasm_vaxprintf(0, fmt, ap);
    va_end(ap);

    return strp;
}
