/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#include "libyuv/rotate.h"
#include "libyuv/rotate_row.h"

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define TANY(NAMEANY, TPOS_SIMD, TPOS_C, MASK)                        \
  void NAMEANY(const uint8_t* src, int src_stride, uint8_t* dst,      \
               int dst_stride, int width) {                           \
    int r = width & MASK;                                             \
    int n = width - r;                                                \
    if (n > 0) {                                                      \
      TPOS_SIMD(src, src_stride, dst, dst_stride, n);                 \
    }                                                                 \
    TPOS_C(src + n, src_stride, dst + n * dst_stride, dst_stride, r); \
  }

#ifdef HAS_TRANSPOSEWX8_NEON
TANY(TransposeWx8_Any_NEON, TransposeWx8_NEON, TransposeWx8_C, 7)
#endif
#ifdef HAS_TRANSPOSEWX16_NEON
TANY(TransposeWx16_Any_NEON, TransposeWx16_NEON, TransposeWx16_C, 15)
#endif
#ifdef HAS_TRANSPOSEWX8_SSSE3
TANY(TransposeWx8_Any_SSSE3, TransposeWx8_SSSE3, TransposeWx8_C, 7)
#endif
#ifdef HAS_TRANSPOSEWX8_FAST_SSSE3
TANY(TransposeWx8_Fast_Any_SSSE3, TransposeWx8_Fast_SSSE3, TransposeWx8_C, 15)
#endif
#ifdef HAS_TRANSPOSEWX16_MSA
TANY(TransposeWx16_Any_MSA, TransposeWx16_MSA, TransposeWx16_C, 15)
#endif
#ifdef HAS_TRANSPOSEWX16_LSX
TANY(TransposeWx16_Any_LSX, TransposeWx16_LSX, TransposeWx16_C, 15)
#endif
#undef TANY

#define TUVANY(NAMEANY, TPOS_SIMD, MASK)                                       \
  void NAMEANY(const uint8_t* src, int src_stride, uint8_t* dst_a,             \
               int dst_stride_a, uint8_t* dst_b, int dst_stride_b,             \
               int width) {                                                    \
    int r = width & MASK;                                                      \
    int n = width - r;                                                         \
    if (n > 0) {                                                               \
      TPOS_SIMD(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b, n); \
    }                                                                          \
    TransposeUVWx8_C(src + n * 2, src_stride, dst_a + n * dst_stride_a,        \
                     dst_stride_a, dst_b + n * dst_stride_b, dst_stride_b, r); \
  }

#ifdef HAS_TRANSPOSEUVWX8_NEON
TUVANY(TransposeUVWx8_Any_NEON, TransposeUVWx8_NEON, 7)
#endif
#ifdef HAS_TRANSPOSEUVWX8_SSE2
TUVANY(TransposeUVWx8_Any_SSE2, TransposeUVWx8_SSE2, 7)
#endif
#ifdef HAS_TRANSPOSEUVWX16_MSA
TUVANY(TransposeUVWx16_Any_MSA, TransposeUVWx16_MSA, 7)
#endif
#ifdef HAS_TRANSPOSEUVWX16_LSX
TUVANY(TransposeUVWx16_Any_LSX, TransposeUVWx16_LSX, 7)
#endif
#undef TUVANY

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
