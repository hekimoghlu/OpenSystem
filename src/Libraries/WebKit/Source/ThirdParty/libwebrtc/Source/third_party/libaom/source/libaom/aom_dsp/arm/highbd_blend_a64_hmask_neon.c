/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/arm/blend_neon.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/blend.h"

void aom_highbd_blend_a64_hmask_neon(uint8_t *dst_8, uint32_t dst_stride,
                                     const uint8_t *src0_8,
                                     uint32_t src0_stride,
                                     const uint8_t *src1_8,
                                     uint32_t src1_stride, const uint8_t *mask,
                                     int w, int h, int bd) {
  (void)bd;

  const uint16_t *src0 = CONVERT_TO_SHORTPTR(src0_8);
  const uint16_t *src1 = CONVERT_TO_SHORTPTR(src1_8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst_8);

  assert(IMPLIES(src0 == dst, src0_stride == dst_stride));
  assert(IMPLIES(src1 == dst, src1_stride == dst_stride));

  assert(h >= 1);
  assert(w >= 1);
  assert(IS_POWER_OF_TWO(h));
  assert(IS_POWER_OF_TWO(w));

  assert(bd == 8 || bd == 10 || bd == 12);

  if (w >= 8) {
    do {
      int i = 0;
      do {
        uint16x8_t m0 = vmovl_u8(vld1_u8(mask + i));
        uint16x8_t s0 = vld1q_u16(src0 + i);
        uint16x8_t s1 = vld1q_u16(src1 + i);

        uint16x8_t blend = alpha_blend_a64_u16x8(m0, s0, s1);

        vst1q_u16(dst + i, blend);
        i += 8;
      } while (i < w);

      src0 += src0_stride;
      src1 += src1_stride;
      dst += dst_stride;
    } while (--h != 0);
  } else if (w == 4) {
    const uint16x8_t m0 = vmovl_u8(load_unaligned_dup_u8_4x2(mask));
    do {
      uint16x8_t s0 = load_unaligned_u16_4x2(src0, src0_stride);
      uint16x8_t s1 = load_unaligned_u16_4x2(src1, src1_stride);

      uint16x8_t blend = alpha_blend_a64_u16x8(m0, s0, s1);

      store_u16x4_strided_x2(dst, dst_stride, blend);

      src0 += 2 * src0_stride;
      src1 += 2 * src1_stride;
      dst += 2 * dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 2 && h >= 8) {
    const uint16x4_t m0 =
        vget_low_u16(vmovl_u8(load_unaligned_dup_u8_2x4(mask)));
    do {
      uint16x4_t s0 = load_unaligned_u16_2x2(src0, src0_stride);
      uint16x4_t s1 = load_unaligned_u16_2x2(src1, src1_stride);

      uint16x4_t blend = alpha_blend_a64_u16x4(m0, s0, s1);

      store_u16x2_strided_x2(dst, dst_stride, blend);

      src0 += 2 * src0_stride;
      src1 += 2 * src1_stride;
      dst += 2 * dst_stride;
      h -= 2;
    } while (h != 0);
  } else {
    aom_highbd_blend_a64_hmask_c(dst_8, dst_stride, src0_8, src0_stride, src1_8,
                                 src1_stride, mask, w, h, bd);
  }
}
