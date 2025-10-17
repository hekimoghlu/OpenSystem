/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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

#include "./vp8_rtcd.h"

void vp8_dc_only_idct_add_neon(int16_t input_dc, unsigned char *pred_ptr,
                               int pred_stride, unsigned char *dst_ptr,
                               int dst_stride) {
  int i;
  uint16_t a1 = ((input_dc + 4) >> 3);
  uint32x2_t d2u32 = vdup_n_u32(0);
  uint8x8_t d2u8;
  uint16x8_t q1u16;
  uint16x8_t qAdd;

  qAdd = vdupq_n_u16(a1);

  for (i = 0; i < 2; ++i) {
    d2u32 = vld1_lane_u32((const uint32_t *)pred_ptr, d2u32, 0);
    pred_ptr += pred_stride;
    d2u32 = vld1_lane_u32((const uint32_t *)pred_ptr, d2u32, 1);
    pred_ptr += pred_stride;

    q1u16 = vaddw_u8(qAdd, vreinterpret_u8_u32(d2u32));
    d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q1u16));

    vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d2u8), 0);
    dst_ptr += dst_stride;
    vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d2u8), 1);
    dst_ptr += dst_stride;
  }
}
