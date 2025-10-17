/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void idct4x4_1_add_kernel(uint8_t **dest, const int stride,
                                        const int16x8_t res,
                                        uint32x2_t *const d) {
  uint16x8_t a;
  uint8x8_t b;
  *d = vld1_lane_u32((const uint32_t *)*dest, *d, 0);
  *d = vld1_lane_u32((const uint32_t *)(*dest + stride), *d, 1);
  a = vaddw_u8(vreinterpretq_u16_s16(res), vreinterpret_u8_u32(*d));
  b = vqmovun_s16(vreinterpretq_s16_u16(a));
  vst1_lane_u32((uint32_t *)*dest, vreinterpret_u32_u8(b), 0);
  *dest += stride;
  vst1_lane_u32((uint32_t *)*dest, vreinterpret_u32_u8(b), 1);
  *dest += stride;
}

void vpx_idct4x4_1_add_neon(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  const int16_t out0 =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));
  const int16_t out1 = WRAPLOW(dct_const_round_shift(out0 * cospi_16_64));
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 4);
  const int16x8_t dc = vdupq_n_s16(a1);
  uint32x2_t d = vdup_n_u32(0);

  assert(!((intptr_t)dest % sizeof(uint32_t)));
  assert(!(stride % sizeof(uint32_t)));

  idct4x4_1_add_kernel(&dest, stride, dc, &d);
  idct4x4_1_add_kernel(&dest, stride, dc, &d);
}
