/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
#ifndef DAV1D_COMMON_BITDEPTH_H
#define DAV1D_COMMON_BITDEPTH_H

#include <stdint.h>
#include <string.h>

#include "common/attributes.h"

#if !defined(BITDEPTH)
typedef void pixel;
typedef void coef;
#define HIGHBD_DECL_SUFFIX /* nothing */
#define HIGHBD_CALL_SUFFIX /* nothing */
#define HIGHBD_TAIL_SUFFIX /* nothing */
#elif BITDEPTH == 8
typedef uint8_t pixel;
typedef int16_t coef;
#define PIXEL_TYPE uint8_t
#define COEF_TYPE int16_t
#define pixel_copy memcpy
#define pixel_set memset
#define iclip_pixel iclip_u8
#define PIX_HEX_FMT "%02x"
#define bitfn(x) x##_8bpc
#define BF(x, suffix) x##_8bpc_##suffix
#define PXSTRIDE(x) (x)
#define highbd_only(x)
#define HIGHBD_DECL_SUFFIX /* nothing */
#define HIGHBD_CALL_SUFFIX /* nothing */
#define HIGHBD_TAIL_SUFFIX /* nothing */
#define bitdepth_from_max(x) 8
#define BITDEPTH_MAX 0xff
#elif BITDEPTH == 16
typedef uint16_t pixel;
typedef int32_t coef;
#define PIXEL_TYPE uint16_t
#define COEF_TYPE int32_t
#define pixel_copy(a, b, c) memcpy(a, b, (c) << 1)
static inline void pixel_set(pixel *const dst, const int val, const int num) {
    for (int n = 0; n < num; n++)
        dst[n] = val;
}
#define PIX_HEX_FMT "%03x"
#define iclip_pixel(x) iclip(x, 0, bitdepth_max)
#define HIGHBD_DECL_SUFFIX , const int bitdepth_max
#define HIGHBD_CALL_SUFFIX , f->bitdepth_max
#define HIGHBD_TAIL_SUFFIX , bitdepth_max
#define bitdepth_from_max(bitdepth_max) (32 - clz(bitdepth_max))
#define BITDEPTH_MAX bitdepth_max
#define bitfn(x) x##_16bpc
#define BF(x, suffix) x##_16bpc_##suffix
static inline ptrdiff_t PXSTRIDE(const ptrdiff_t x) {
    assert(!(x & 1));
    return x >> 1;
}
#define highbd_only(x) x
#else
#error invalid value for bitdepth
#endif
#define bytefn(x) bitfn(x)

#define bitfn_decls(name, ...) \
name##_8bpc(__VA_ARGS__); \
name##_16bpc(__VA_ARGS__)

#endif /* DAV1D_COMMON_BITDEPTH_H */
