/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#ifndef DAV1D_INPUT_PARSE_H
#define DAV1D_INPUT_PARSE_H

#include <limits.h>

#include "dav1d/headers.h"

static int leb128(FILE *const f, size_t *const len) {
    uint64_t val = 0;
    unsigned i = 0, more;
    do {
        uint8_t v;
        if (fread(&v, 1, 1, f) < 1)
            return -1;
        more = v & 0x80;
        val |= ((uint64_t) (v & 0x7F)) << (i * 7);
        i++;
    } while (more && i < 8);
    if (val > UINT_MAX || more)
        return -1;
    *len = (size_t) val;
    return i;
}

// these functions are based on an implementation from FFmpeg, and relicensed
// with author's permission

static int leb(const uint8_t *ptr, int sz, size_t *const len) {
    uint64_t val = 0;
    unsigned i = 0, more;
    do {
        if (!sz--) return -1;
        const int v = *ptr++;
        more = v & 0x80;
        val |= ((uint64_t) (v & 0x7F)) << (i * 7);
        i++;
    } while (more && i < 8);
    if (val > UINT_MAX || more)
        return -1;
    *len = (size_t) val;
    return i;
}

static inline int parse_obu_header(const uint8_t *buf, int buf_size,
                                   size_t *const obu_size,
                                   enum Dav1dObuType *const type,
                                   const int allow_implicit_size)
{
    int ret, extension_flag, has_size_flag;

    if (!buf_size)
        return -1;
    if (*buf & 0x80) // obu_forbidden_bit
        return -1;

    *type = (*buf & 0x78) >> 3;
    extension_flag = (*buf & 0x4) >> 2;
    has_size_flag  = (*buf & 0x2) >> 1;
    // ignore obu_reserved_1bit
    buf++;
    buf_size--;

    if (extension_flag) {
        buf++;
        buf_size--;
        // ignore fields
    }

    if (has_size_flag) {
        ret = leb(buf, buf_size, obu_size);
        if (ret < 0)
            return -1;
        return (int) *obu_size + ret + 1 + extension_flag;
    } else if (!allow_implicit_size)
        return -1;

    *obu_size = buf_size;
    return buf_size + 1 + extension_flag;
}

#endif /* DAV1D_INPUT_PARSE_H */
