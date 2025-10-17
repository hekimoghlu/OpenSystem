/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
#include <assert.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/bitdepth_conversion_sse2.h"
#include "vpx_dsp/x86/quantize_sse2.h"
#include "vp9/common/vp9_scan.h"

void vpx_quantize_b_sse2(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                         const struct macroblock_plane *const mb_plane,
                         tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                         const int16_t *dequant_ptr, uint16_t *eob_ptr,
                         const struct ScanOrder *const scan_order) {
  const __m128i zero = _mm_setzero_si128();
  int index = 16;
  const int16_t *iscan = scan_order->iscan;

  __m128i zbin, round, quant, dequant, shift;
  __m128i coeff0, coeff1, coeff0_sign, coeff1_sign;
  __m128i qcoeff0, qcoeff1;
  __m128i cmp_mask0, cmp_mask1;
  __m128i eob, eob0;

  // Setup global values.
  load_b_values(mb_plane, &zbin, &round, &quant, dequant_ptr, &dequant, &shift);

  // Do DC and first 15 AC.
  coeff0 = load_tran_low(coeff_ptr);
  coeff1 = load_tran_low(coeff_ptr + 8);

  // Poor man's abs().
  coeff0_sign = _mm_srai_epi16(coeff0, 15);
  coeff1_sign = _mm_srai_epi16(coeff1, 15);
  qcoeff0 = invert_sign_sse2(coeff0, coeff0_sign);
  qcoeff1 = invert_sign_sse2(coeff1, coeff1_sign);

  cmp_mask0 = _mm_cmpgt_epi16(qcoeff0, zbin);
  zbin = _mm_unpackhi_epi64(zbin, zbin);  // Switch DC to AC
  cmp_mask1 = _mm_cmpgt_epi16(qcoeff1, zbin);

  calculate_qcoeff(&qcoeff0, round, quant, shift);

  round = _mm_unpackhi_epi64(round, round);
  quant = _mm_unpackhi_epi64(quant, quant);
  shift = _mm_unpackhi_epi64(shift, shift);

  calculate_qcoeff(&qcoeff1, round, quant, shift);

  // Reinsert signs
  qcoeff0 = invert_sign_sse2(qcoeff0, coeff0_sign);
  qcoeff1 = invert_sign_sse2(qcoeff1, coeff1_sign);

  // Mask out zbin threshold coeffs
  qcoeff0 = _mm_and_si128(qcoeff0, cmp_mask0);
  qcoeff1 = _mm_and_si128(qcoeff1, cmp_mask1);

  store_tran_low(qcoeff0, qcoeff_ptr);
  store_tran_low(qcoeff1, qcoeff_ptr + 8);

  calculate_dqcoeff_and_store(qcoeff0, dequant, dqcoeff_ptr);
  dequant = _mm_unpackhi_epi64(dequant, dequant);
  calculate_dqcoeff_and_store(qcoeff1, dequant, dqcoeff_ptr + 8);

  eob = scan_for_eob(&qcoeff0, &qcoeff1, iscan, 0, zero);

  // AC only loop.
  while (index < n_coeffs) {
    coeff0 = load_tran_low(coeff_ptr + index);
    coeff1 = load_tran_low(coeff_ptr + index + 8);

    coeff0_sign = _mm_srai_epi16(coeff0, 15);
    coeff1_sign = _mm_srai_epi16(coeff1, 15);
    qcoeff0 = invert_sign_sse2(coeff0, coeff0_sign);
    qcoeff1 = invert_sign_sse2(coeff1, coeff1_sign);

    cmp_mask0 = _mm_cmpgt_epi16(qcoeff0, zbin);
    cmp_mask1 = _mm_cmpgt_epi16(qcoeff1, zbin);

    calculate_qcoeff(&qcoeff0, round, quant, shift);
    calculate_qcoeff(&qcoeff1, round, quant, shift);

    qcoeff0 = invert_sign_sse2(qcoeff0, coeff0_sign);
    qcoeff1 = invert_sign_sse2(qcoeff1, coeff1_sign);

    qcoeff0 = _mm_and_si128(qcoeff0, cmp_mask0);
    qcoeff1 = _mm_and_si128(qcoeff1, cmp_mask1);

    store_tran_low(qcoeff0, qcoeff_ptr + index);
    store_tran_low(qcoeff1, qcoeff_ptr + index + 8);

    calculate_dqcoeff_and_store(qcoeff0, dequant, dqcoeff_ptr + index);
    calculate_dqcoeff_and_store(qcoeff1, dequant, dqcoeff_ptr + index + 8);

    eob0 = scan_for_eob(&qcoeff0, &qcoeff1, iscan, index, zero);
    eob = _mm_max_epi16(eob, eob0);

    index += 16;
  }

  *eob_ptr = accumulate_eob(eob);
}
