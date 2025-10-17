/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
#ifndef AOM_AV1_COMMON_X86_AV1_TXFM_SSE4_H_
#define AOM_AV1_COMMON_X86_AV1_TXFM_SSE4_H_

#include <smmintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline __m128i av1_round_shift_32_sse4_1(__m128i vec, int bit) {
  __m128i tmp, round;
  round = _mm_set1_epi32(1 << (bit - 1));
  tmp = _mm_add_epi32(vec, round);
  return _mm_srai_epi32(tmp, bit);
}

static inline void av1_round_shift_array_32_sse4_1(const __m128i *input,
                                                   __m128i *output,
                                                   const int size,
                                                   const int bit) {
  if (bit > 0) {
    int i;
    for (i = 0; i < size; i++) {
      output[i] = av1_round_shift_32_sse4_1(input[i], bit);
    }
  } else {
    int i;
    for (i = 0; i < size; i++) {
      output[i] = _mm_slli_epi32(input[i], -bit);
    }
  }
}

static inline void av1_round_shift_rect_array_32_sse4_1(const __m128i *input,
                                                        __m128i *output,
                                                        const int size,
                                                        const int bit,
                                                        const int val) {
  const __m128i sqrt2 = _mm_set1_epi32(val);
  if (bit > 0) {
    int i;
    for (i = 0; i < size; i++) {
      const __m128i r0 = av1_round_shift_32_sse4_1(input[i], bit);
      const __m128i r1 = _mm_mullo_epi32(sqrt2, r0);
      output[i] = av1_round_shift_32_sse4_1(r1, NewSqrt2Bits);
    }
  } else {
    int i;
    for (i = 0; i < size; i++) {
      const __m128i r0 = _mm_slli_epi32(input[i], -bit);
      const __m128i r1 = _mm_mullo_epi32(sqrt2, r0);
      output[i] = av1_round_shift_32_sse4_1(r1, NewSqrt2Bits);
    }
  }
}

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_COMMON_X86_AV1_TXFM_SSE4_H_
