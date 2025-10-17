/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include <emmintrin.h>  // SSE2

#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"

static inline __m128i reduce_sum_epi64(__m128i reg) {
  __m128i reg_hi = _mm_srli_si128(reg, 8);
  reg = _mm_add_epi64(reg, reg_hi);

  return reg;
}

int64_t av1_block_error_lp_sse2(const int16_t *coeff, const int16_t *dqcoeff,
                                intptr_t block_size) {
  assert(block_size % 16 == 0);
  assert(block_size >= 16);

  const __m128i zero = _mm_setzero_si128();
  __m128i accum_0 = zero;
  __m128i accum_1 = zero;

  for (int i = 0; i < block_size; i += 16) {
    // Load 8 elements for coeff and dqcoeff.
    const __m128i _coeff_0 = _mm_loadu_si128((const __m128i *)coeff);
    const __m128i _coeff_1 = _mm_loadu_si128((const __m128i *)(coeff + 8));
    const __m128i _dqcoeff_0 = _mm_loadu_si128((const __m128i *)dqcoeff);
    const __m128i _dqcoeff_1 = _mm_loadu_si128((const __m128i *)(dqcoeff + 8));
    // Compute the diff
    const __m128i diff_0 = _mm_sub_epi16(_dqcoeff_0, _coeff_0);
    const __m128i diff_1 = _mm_sub_epi16(_dqcoeff_1, _coeff_1);
    // Compute the error
    const __m128i error_0 = _mm_madd_epi16(diff_0, diff_0);
    const __m128i error_1 = _mm_madd_epi16(diff_1, diff_1);

    const __m128i error_lo_0 = _mm_unpacklo_epi32(error_0, zero);
    const __m128i error_lo_1 = _mm_unpacklo_epi32(error_1, zero);
    const __m128i error_hi_0 = _mm_unpackhi_epi32(error_0, zero);
    const __m128i error_hi_1 = _mm_unpackhi_epi32(error_1, zero);

    // Accumulate
    accum_0 = _mm_add_epi64(accum_0, error_lo_0);
    accum_1 = _mm_add_epi64(accum_1, error_lo_1);
    accum_0 = _mm_add_epi64(accum_0, error_hi_0);
    accum_1 = _mm_add_epi64(accum_1, error_hi_1);

    // Advance
    coeff += 16;
    dqcoeff += 16;
  }

  __m128i accum = _mm_add_epi64(accum_0, accum_1);
  // Reduce sum the register
  accum = reduce_sum_epi64(accum);

  // Store the results.
#if AOM_ARCH_X86_64
  return _mm_cvtsi128_si64(accum);
#else
  int64_t result;
  _mm_storel_epi64((__m128i *)&result, accum);
  return result;
#endif  // AOM_ARCH_X86_64
}
