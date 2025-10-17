/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#ifndef VPX_VPX_DSP_X86_QUANTIZE_SSSE3_H_
#define VPX_VPX_DSP_X86_QUANTIZE_SSSE3_H_

#include <emmintrin.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/quantize_sse2.h"

static INLINE void calculate_dqcoeff_and_store_32x32(const __m128i qcoeff,
                                                     const __m128i dequant,
                                                     const __m128i zero,
                                                     tran_low_t *dqcoeff) {
  // Un-sign to bias rounding like C.
  const __m128i coeff = _mm_abs_epi16(qcoeff);

  const __m128i sign_0 = _mm_unpacklo_epi16(zero, qcoeff);
  const __m128i sign_1 = _mm_unpackhi_epi16(zero, qcoeff);

  const __m128i low = _mm_mullo_epi16(coeff, dequant);
  const __m128i high = _mm_mulhi_epi16(coeff, dequant);
  __m128i dqcoeff32_0 = _mm_unpacklo_epi16(low, high);
  __m128i dqcoeff32_1 = _mm_unpackhi_epi16(low, high);

  // "Divide" by 2.
  dqcoeff32_0 = _mm_srli_epi32(dqcoeff32_0, 1);
  dqcoeff32_1 = _mm_srli_epi32(dqcoeff32_1, 1);

  dqcoeff32_0 = _mm_sign_epi32(dqcoeff32_0, sign_0);
  dqcoeff32_1 = _mm_sign_epi32(dqcoeff32_1, sign_1);

#if CONFIG_VP9_HIGHBITDEPTH
  _mm_store_si128((__m128i *)(dqcoeff), dqcoeff32_0);
  _mm_store_si128((__m128i *)(dqcoeff + 4), dqcoeff32_1);
#else
  _mm_store_si128((__m128i *)(dqcoeff),
                  _mm_packs_epi32(dqcoeff32_0, dqcoeff32_1));
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

#endif  // VPX_VPX_DSP_X86_QUANTIZE_SSSE3_H_
