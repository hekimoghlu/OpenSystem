/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE uint8x8_t create_dcd(const int16_t dc) {
  int16x8_t t = vdupq_n_s16(dc);
  return vqmovun_s16(t);
}

static INLINE void idct8x8_1_add_pos_kernel(uint8_t **dest, const int stride,
                                            const uint8x8_t res) {
  const uint8x8_t a = vld1_u8(*dest);
  const uint8x8_t b = vqadd_u8(a, res);
  vst1_u8(*dest, b);
  *dest += stride;
}

static INLINE void idct8x8_1_add_neg_kernel(uint8_t **dest, const int stride,
                                            const uint8x8_t res) {
  const uint8x8_t a = vld1_u8(*dest);
  const uint8x8_t b = vqsub_u8(a, res);
  vst1_u8(*dest, b);
  *dest += stride;
}

void vpx_idct8x8_1_add_neon(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  const int16_t out0 =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));
  const int16_t out1 = WRAPLOW(dct_const_round_shift(out0 * cospi_16_64));
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 5);

  if (a1 >= 0) {
    const uint8x8_t dc = create_dcd(a1);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
    idct8x8_1_add_pos_kernel(&dest, stride, dc);
  } else {
    const uint8x8_t dc = create_dcd(-a1);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
    idct8x8_1_add_neg_kernel(&dest, stride, dc);
  }
}
