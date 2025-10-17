/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#ifndef DAV1D_COMMON_INTOPS_H
#define DAV1D_COMMON_INTOPS_H

#include <stdint.h>

#include "common/attributes.h"

static inline int imax(const int a, const int b) {
    return a > b ? a : b;
}

static inline int imin(const int a, const int b) {
    return a < b ? a : b;
}

static inline unsigned umax(const unsigned a, const unsigned b) {
    return a > b ? a : b;
}

static inline unsigned umin(const unsigned a, const unsigned b) {
    return a < b ? a : b;
}

static inline int iclip(const int v, const int min, const int max) {
    return v < min ? min : v > max ? max : v;
}

static inline int iclip_u8(const int v) {
    return iclip(v, 0, 255);
}

static inline int apply_sign(const int v, const int s) {
    return s < 0 ? -v : v;
}

static inline int apply_sign64(const int v, const int64_t s) {
    return s < 0 ? -v : v;
}

static inline int ulog2(const unsigned v) {
    return 31 - clz(v);
}

static inline int u64log2(const uint64_t v) {
    return 63 - clzll(v);
}

static inline unsigned inv_recenter(const unsigned r, const unsigned v) {
    if (v > (r << 1))
        return v;
    else if ((v & 1) == 0)
        return (v >> 1) + r;
    else
        return r - ((v + 1) >> 1);
}

#endif /* DAV1D_COMMON_INTOPS_H */
