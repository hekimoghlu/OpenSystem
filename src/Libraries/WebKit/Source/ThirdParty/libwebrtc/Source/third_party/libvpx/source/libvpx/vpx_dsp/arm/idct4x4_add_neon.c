/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/txfm_common.h"

void vpx_idct4x4_16_add_neon(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const uint8_t *dst = dest;
  uint32x2_t s32 = vdup_n_u32(0);
  int16x8_t a[2];
  uint8x8_t s, d[2];
  uint16x8_t sum[2];

  assert(!((intptr_t)dest % sizeof(uint32_t)));
  assert(!(stride % sizeof(uint32_t)));

  // Rows
  a[0] = load_tran_low_to_s16q(input);
  a[1] = load_tran_low_to_s16q(input + 8);
  transpose_idct4x4_16_bd8(a);

  // Columns
  a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
  transpose_idct4x4_16_bd8(a);
  a[0] = vrshrq_n_s16(a[0], 4);
  a[1] = vrshrq_n_s16(a[1], 4);

  s = load_u8(dst, stride);
  dst += 2 * stride;
  // The elements are loaded in reverse order.
  s32 = vld1_lane_u32((const uint32_t *)dst, s32, 1);
  dst += stride;
  s32 = vld1_lane_u32((const uint32_t *)dst, s32, 0);

  sum[0] = vaddw_u8(vreinterpretq_u16_s16(a[0]), s);
  sum[1] = vaddw_u8(vreinterpretq_u16_s16(a[1]), vreinterpret_u8_u32(s32));
  d[0] = vqmovun_s16(vreinterpretq_s16_u16(sum[0]));
  d[1] = vqmovun_s16(vreinterpretq_s16_u16(sum[1]));

  store_u8(dest, stride, d[0]);
  dest += 2 * stride;
  // The elements are stored in reverse order.
  vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d[1]), 1);
  dest += stride;
  vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d[1]), 0);
}
