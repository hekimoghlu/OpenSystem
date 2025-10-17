/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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

#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/blend_neon.h"
#include "aom_dsp/arm/mem_neon.h"

void aom_blend_a64_hmask_neon(uint8_t *dst, uint32_t dst_stride,
                              const uint8_t *src0, uint32_t src0_stride,
                              const uint8_t *src1, uint32_t src1_stride,
                              const uint8_t *mask, int w, int h) {
  assert(IMPLIES(src0 == dst, src0_stride == dst_stride));
  assert(IMPLIES(src1 == dst, src1_stride == dst_stride));

  assert(h >= 2);
  assert(w >= 2);
  assert(IS_POWER_OF_TWO(h));
  assert(IS_POWER_OF_TWO(w));

  if (w > 8) {
    do {
      int i = 0;
      do {
        uint8x16_t m0 = vld1q_u8(mask + i);
        uint8x16_t s0 = vld1q_u8(src0 + i);
        uint8x16_t s1 = vld1q_u8(src1 + i);

        uint8x16_t blend = alpha_blend_a64_u8x16(m0, s0, s1);

        vst1q_u8(dst + i, blend);

        i += 16;
      } while (i < w);

      src0 += src0_stride;
      src1 += src1_stride;
      dst += dst_stride;
    } while (--h != 0);
  } else if (w == 8) {
    const uint8x8_t m0 = vld1_u8(mask);
    do {
      uint8x8_t s0 = vld1_u8(src0);
      uint8x8_t s1 = vld1_u8(src1);

      uint8x8_t blend = alpha_blend_a64_u8x8(m0, s0, s1);

      vst1_u8(dst, blend);

      src0 += src0_stride;
      src1 += src1_stride;
      dst += dst_stride;
    } while (--h != 0);
  } else if (w == 4) {
    const uint8x8_t m0 = load_unaligned_dup_u8_4x2(mask);
    do {
      uint8x8_t s0 = load_unaligned_u8_4x2(src0, src0_stride);
      uint8x8_t s1 = load_unaligned_u8_4x2(src1, src1_stride);

      uint8x8_t blend = alpha_blend_a64_u8x8(m0, s0, s1);

      store_u8x4_strided_x2(dst, dst_stride, blend);

      src0 += 2 * src0_stride;
      src1 += 2 * src1_stride;
      dst += 2 * dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 2 && h >= 16) {
    const uint8x8_t m0 = vreinterpret_u8_u16(vld1_dup_u16((uint16_t *)mask));
    do {
      uint8x8_t s0 = load_unaligned_u8_2x2(src0, src0_stride);
      uint8x8_t s1 = load_unaligned_u8_2x2(src1, src1_stride);

      uint8x8_t blend = alpha_blend_a64_u8x8(m0, s0, s1);

      store_u8x2_strided_x2(dst, dst_stride, blend);

      src0 += 2 * src0_stride;
      src1 += 2 * src1_stride;
      dst += 2 * dst_stride;
      h -= 2;
    } while (h != 0);
  } else {
    aom_blend_a64_hmask_c(dst, dst_stride, src0, src0_stride, src1, src1_stride,
                          mask, w, h);
  }
}
