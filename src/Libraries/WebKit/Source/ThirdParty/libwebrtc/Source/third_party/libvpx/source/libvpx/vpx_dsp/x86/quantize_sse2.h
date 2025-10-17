/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef VPX_VPX_DSP_X86_QUANTIZE_SSE2_H_
#define VPX_VPX_DSP_X86_QUANTIZE_SSE2_H_

#include <emmintrin.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vp9/encoder/vp9_block.h"

static INLINE void load_b_values(const struct macroblock_plane *const mb_plane,
                                 __m128i *zbin, __m128i *round, __m128i *quant,
                                 const int16_t *dequant_ptr, __m128i *dequant,
                                 __m128i *shift) {
  *zbin = _mm_load_si128((const __m128i *)mb_plane->zbin);
  *round = _mm_load_si128((const __m128i *)mb_plane->round);
  *quant = _mm_load_si128((const __m128i *)mb_plane->quant);
  *zbin = _mm_sub_epi16(*zbin, _mm_set1_epi16(1));
  *dequant = _mm_load_si128((const __m128i *)dequant_ptr);
  *shift = _mm_load_si128((const __m128i *)mb_plane->quant_shift);
}

static INLINE void load_b_values32x32(
    const struct macroblock_plane *const mb_plane, __m128i *zbin,
    __m128i *round, __m128i *quant, const int16_t *dequant_ptr,
    __m128i *dequant, __m128i *shift) {
  const __m128i one = _mm_set1_epi16(1);
  // The 32x32 halves zbin and round.
  *zbin = _mm_load_si128((const __m128i *)mb_plane->zbin);
  // Shift with rounding.
  *zbin = _mm_add_epi16(*zbin, one);
  *zbin = _mm_srli_epi16(*zbin, 1);
  // x86 has no "greater *or equal*" comparison. Subtract 1 from zbin so
  // it is a strict "greater" comparison.
  *zbin = _mm_sub_epi16(*zbin, one);

  *round = _mm_load_si128((const __m128i *)mb_plane->round);
  *round = _mm_add_epi16(*round, one);
  *round = _mm_srli_epi16(*round, 1);

  *quant = _mm_load_si128((const __m128i *)mb_plane->quant);
  *dequant = _mm_load_si128((const __m128i *)dequant_ptr);
  *shift = _mm_load_si128((const __m128i *)mb_plane->quant_shift);
  // I suspect this is not technically OK because quant_shift can be up
  // to 1 << 16 and shifting up again will outrange that, but the test is not
  // comprehensive enough to catch that and "it's been that way forever"
  *shift = _mm_slli_epi16(*shift, 1);
}

static INLINE void load_fp_values(const struct macroblock_plane *mb_plane,
                                  __m128i *round, __m128i *quant,
                                  const int16_t *dequant_ptr,
                                  __m128i *dequant) {
  *round = _mm_load_si128((const __m128i *)mb_plane->round_fp);
  *quant = _mm_load_si128((const __m128i *)mb_plane->quant_fp);
  *dequant = _mm_load_si128((const __m128i *)dequant_ptr);
}

// With ssse3 and later abs() and sign() are preferred.
static INLINE __m128i invert_sign_sse2(__m128i a, __m128i sign) {
  a = _mm_xor_si128(a, sign);
  return _mm_sub_epi16(a, sign);
}

static INLINE void calculate_qcoeff(__m128i *coeff, const __m128i round,
                                    const __m128i quant, const __m128i shift) {
  __m128i tmp, qcoeff;
  qcoeff = _mm_adds_epi16(*coeff, round);
  tmp = _mm_mulhi_epi16(qcoeff, quant);
  qcoeff = _mm_add_epi16(tmp, qcoeff);
  *coeff = _mm_mulhi_epi16(qcoeff, shift);
}

static INLINE void calculate_dqcoeff_and_store(__m128i qcoeff, __m128i dequant,
                                               tran_low_t *dqcoeff) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m128i low = _mm_mullo_epi16(qcoeff, dequant);
  const __m128i high = _mm_mulhi_epi16(qcoeff, dequant);

  const __m128i dqcoeff32_0 = _mm_unpacklo_epi16(low, high);
  const __m128i dqcoeff32_1 = _mm_unpackhi_epi16(low, high);

  _mm_store_si128((__m128i *)(dqcoeff), dqcoeff32_0);
  _mm_store_si128((__m128i *)(dqcoeff + 4), dqcoeff32_1);
#else
  const __m128i dqcoeff16 = _mm_mullo_epi16(qcoeff, dequant);

  _mm_store_si128((__m128i *)(dqcoeff), dqcoeff16);
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

// Scan 16 values for eob reference in scan.
static INLINE __m128i scan_for_eob(__m128i *coeff0, __m128i *coeff1,
                                   const int16_t *scan, const int index,
                                   const __m128i zero) {
  const __m128i zero_coeff0 = _mm_cmpeq_epi16(*coeff0, zero);
  const __m128i zero_coeff1 = _mm_cmpeq_epi16(*coeff1, zero);
  __m128i scan0 = _mm_load_si128((const __m128i *)(scan + index));
  __m128i scan1 = _mm_load_si128((const __m128i *)(scan + index + 8));
  __m128i eob0, eob1;
  eob0 = _mm_andnot_si128(zero_coeff0, scan0);
  eob1 = _mm_andnot_si128(zero_coeff1, scan1);
  return _mm_max_epi16(eob0, eob1);
}

static INLINE int16_t accumulate_eob(__m128i eob) {
  __m128i eob_shuffled;
  eob_shuffled = _mm_shuffle_epi32(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0x1);
  eob = _mm_max_epi16(eob, eob_shuffled);
  return _mm_extract_epi16(eob, 1);
}

#endif  // VPX_VPX_DSP_X86_QUANTIZE_SSE2_H_
