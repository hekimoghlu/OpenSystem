/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void idct32x32_1_add_pos_kernel(uint8_t **dest, const int stride,
                                              const uint8x16_t res) {
  const uint8x16_t a0 = vld1q_u8(*dest);
  const uint8x16_t a1 = vld1q_u8(*dest + 16);
  const uint8x16_t b0 = vqaddq_u8(a0, res);
  const uint8x16_t b1 = vqaddq_u8(a1, res);
  vst1q_u8(*dest, b0);
  vst1q_u8(*dest + 16, b1);
  *dest += stride;
}

static INLINE void idct32x32_1_add_neg_kernel(uint8_t **dest, const int stride,
                                              const uint8x16_t res) {
  const uint8x16_t a0 = vld1q_u8(*dest);
  const uint8x16_t a1 = vld1q_u8(*dest + 16);
  const uint8x16_t b0 = vqsubq_u8(a0, res);
  const uint8x16_t b1 = vqsubq_u8(a1, res);
  vst1q_u8(*dest, b0);
  vst1q_u8(*dest + 16, b1);
  *dest += stride;
}

void vpx_idct32x32_1_add_neon(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  int i;
  const int16_t out0 =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));
  const int16_t out1 = WRAPLOW(dct_const_round_shift(out0 * cospi_16_64));
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 6);

  if (a1 >= 0) {
    const uint8x16_t dc = create_dcq(a1);
    for (i = 0; i < 32; i++) {
      idct32x32_1_add_pos_kernel(&dest, stride, dc);
    }
  } else {
    const uint8x16_t dc = create_dcq(-a1);
    for (i = 0; i < 32; i++) {
      idct32x32_1_add_neg_kernel(&dest, stride, dc);
    }
  }
}
