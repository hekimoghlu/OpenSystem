/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#ifndef VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_AVX2_H_
#define VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_AVX2_H_

#include <immintrin.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

// Load 16 16 bit values. If the source is 32 bits then pack down with
// saturation.
static INLINE __m256i load_tran_low(const tran_low_t *a) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m256i a_low = _mm256_loadu_si256((const __m256i *)a);
  const __m256i a_high = _mm256_loadu_si256((const __m256i *)(a + 8));
  return _mm256_packs_epi32(a_low, a_high);
#else
  return _mm256_loadu_si256((const __m256i *)a);
#endif
}

static INLINE void store_tran_low(__m256i a, tran_low_t *b) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m256i one = _mm256_set1_epi16(1);
  const __m256i a_hi = _mm256_mulhi_epi16(a, one);
  const __m256i a_lo = _mm256_mullo_epi16(a, one);
  const __m256i a_1 = _mm256_unpacklo_epi16(a_lo, a_hi);
  const __m256i a_2 = _mm256_unpackhi_epi16(a_lo, a_hi);
  _mm256_storeu_si256((__m256i *)b, a_1);
  _mm256_storeu_si256((__m256i *)(b + 8), a_2);
#else
  _mm256_storeu_si256((__m256i *)b, a);
#endif
}
#endif  // VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_AVX2_H_
