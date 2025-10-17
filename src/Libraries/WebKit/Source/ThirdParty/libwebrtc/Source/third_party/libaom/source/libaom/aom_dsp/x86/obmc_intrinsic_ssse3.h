/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#ifndef AOM_AOM_DSP_X86_OBMC_INTRINSIC_SSSE3_H_
#define AOM_AOM_DSP_X86_OBMC_INTRINSIC_SSSE3_H_

#include <immintrin.h>

#include "config/aom_config.h"

static inline int32_t xx_hsum_epi32_si32(__m128i v_d) {
  v_d = _mm_hadd_epi32(v_d, v_d);
  v_d = _mm_hadd_epi32(v_d, v_d);
  return _mm_cvtsi128_si32(v_d);
}

static inline int64_t xx_hsum_epi64_si64(__m128i v_q) {
  v_q = _mm_add_epi64(v_q, _mm_srli_si128(v_q, 8));
#if AOM_ARCH_X86_64
  return _mm_cvtsi128_si64(v_q);
#else
  {
    int64_t tmp;
    _mm_storel_epi64((__m128i *)&tmp, v_q);
    return tmp;
  }
#endif
}

static inline int64_t xx_hsum_epi32_si64(__m128i v_d) {
  const __m128i v_sign_d = _mm_cmplt_epi32(v_d, _mm_setzero_si128());
  const __m128i v_0_q = _mm_unpacklo_epi32(v_d, v_sign_d);
  const __m128i v_1_q = _mm_unpackhi_epi32(v_d, v_sign_d);
  return xx_hsum_epi64_si64(_mm_add_epi64(v_0_q, v_1_q));
}

// This is equivalent to ROUND_POWER_OF_TWO_SIGNED(v_val_d, bits)
static inline __m128i xx_roundn_epi32(__m128i v_val_d, int bits) {
  const __m128i v_bias_d = _mm_set1_epi32((1 << bits) >> 1);
  const __m128i v_sign_d = _mm_srai_epi32(v_val_d, 31);
  const __m128i v_tmp_d =
      _mm_add_epi32(_mm_add_epi32(v_val_d, v_bias_d), v_sign_d);
  return _mm_srai_epi32(v_tmp_d, bits);
}

#endif  // AOM_AOM_DSP_X86_OBMC_INTRINSIC_SSSE3_H_
