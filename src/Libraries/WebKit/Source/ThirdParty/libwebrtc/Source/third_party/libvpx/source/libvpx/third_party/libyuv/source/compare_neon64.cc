/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#include "libyuv/basic_types.h"

#include "libyuv/compare_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

// 256 bits at a time
// uses short accumulator which restricts count to 131 KB
uint32_t HammingDistance_NEON(const uint8_t* src_a,
                              const uint8_t* src_b,
                              int count) {
  uint32_t diff;
  asm volatile(
      "movi       v4.8h, #0                      \n"

      "1:                                        \n"
      "ld1        {v0.16b, v1.16b}, [%0], #32    \n"
      "ld1        {v2.16b, v3.16b}, [%1], #32    \n"
      "eor        v0.16b, v0.16b, v2.16b         \n"
      "eor        v1.16b, v1.16b, v3.16b         \n"
      "cnt        v0.16b, v0.16b                 \n"
      "cnt        v1.16b, v1.16b                 \n"
      "subs       %w2, %w2, #32                  \n"
      "add        v0.16b, v0.16b, v1.16b         \n"
      "uadalp     v4.8h, v0.16b                  \n"
      "b.gt       1b                             \n"

      "uaddlv     s4, v4.8h                      \n"
      "fmov       %w3, s4                        \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(diff)
      :
      : "cc", "v0", "v1", "v2", "v3", "v4");
  return diff;
}

uint32_t SumSquareError_NEON(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count) {
  uint32_t sse;
  asm volatile(
      "eor        v16.16b, v16.16b, v16.16b      \n"
      "eor        v18.16b, v18.16b, v18.16b      \n"
      "eor        v17.16b, v17.16b, v17.16b      \n"
      "eor        v19.16b, v19.16b, v19.16b      \n"

      "1:                                        \n"
      "ld1        {v0.16b}, [%0], #16            \n"
      "ld1        {v1.16b}, [%1], #16            \n"
      "subs       %w2, %w2, #16                  \n"
      "usubl      v2.8h, v0.8b, v1.8b            \n"
      "usubl2     v3.8h, v0.16b, v1.16b          \n"
      "smlal      v16.4s, v2.4h, v2.4h           \n"
      "smlal      v17.4s, v3.4h, v3.4h           \n"
      "smlal2     v18.4s, v2.8h, v2.8h           \n"
      "smlal2     v19.4s, v3.8h, v3.8h           \n"
      "b.gt       1b                             \n"

      "add        v16.4s, v16.4s, v17.4s         \n"
      "add        v18.4s, v18.4s, v19.4s         \n"
      "add        v19.4s, v16.4s, v18.4s         \n"
      "addv       s0, v19.4s                     \n"
      "fmov       %w3, s0                        \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(sse)
      :
      : "cc", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19");
  return sse;
}

#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
