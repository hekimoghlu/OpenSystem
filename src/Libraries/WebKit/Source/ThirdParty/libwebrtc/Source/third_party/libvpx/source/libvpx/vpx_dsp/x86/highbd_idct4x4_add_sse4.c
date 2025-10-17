/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#include <smmintrin.h>  // SSE4.1

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/highbd_inv_txfm_sse2.h"
#include "vpx_dsp/x86/highbd_inv_txfm_sse4.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"

void vpx_highbd_idct4x4_16_add_sse4_1(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  __m128i io[4];

  io[0] = _mm_load_si128((const __m128i *)(input + 0));
  io[1] = _mm_load_si128((const __m128i *)(input + 4));
  io[2] = _mm_load_si128((const __m128i *)(input + 8));
  io[3] = _mm_load_si128((const __m128i *)(input + 12));

  if (bd == 8) {
    __m128i io_short[2];

    io_short[0] = _mm_packs_epi32(io[0], io[1]);
    io_short[1] = _mm_packs_epi32(io[2], io[3]);
    idct4_sse2(io_short);
    idct4_sse2(io_short);
    io_short[0] = _mm_add_epi16(io_short[0], _mm_set1_epi16(8));
    io_short[1] = _mm_add_epi16(io_short[1], _mm_set1_epi16(8));
    io[0] = _mm_srai_epi16(io_short[0], 4);
    io[1] = _mm_srai_epi16(io_short[1], 4);
  } else {
    highbd_idct4_sse4_1(io);
    highbd_idct4_sse4_1(io);
    io[0] = wraplow_16bit_shift4(io[0], io[1], _mm_set1_epi32(8));
    io[1] = wraplow_16bit_shift4(io[2], io[3], _mm_set1_epi32(8));
  }

  recon_and_store_4x4(io, dest, stride, bd);
}
