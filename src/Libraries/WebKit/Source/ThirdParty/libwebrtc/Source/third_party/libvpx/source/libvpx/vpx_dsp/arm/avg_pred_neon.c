/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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

void vpx_comp_avg_pred_neon(uint8_t *comp, const uint8_t *pred, int width,
                            int height, const uint8_t *ref, int ref_stride) {
  if (width > 8) {
    int x, y = height;
    do {
      for (x = 0; x < width; x += 16) {
        const uint8x16_t p = vld1q_u8(pred + x);
        const uint8x16_t r = vld1q_u8(ref + x);
        const uint8x16_t avg = vrhaddq_u8(p, r);
        vst1q_u8(comp + x, avg);
      }
      comp += width;
      pred += width;
      ref += ref_stride;
    } while (--y);
  } else if (width == 8) {
    int i = width * height;
    do {
      const uint8x16_t p = vld1q_u8(pred);
      uint8x16_t r;
      const uint8x8_t r_0 = vld1_u8(ref);
      const uint8x8_t r_1 = vld1_u8(ref + ref_stride);
      r = vcombine_u8(r_0, r_1);
      ref += 2 * ref_stride;
      r = vrhaddq_u8(r, p);
      vst1q_u8(comp, r);

      pred += 16;
      comp += 16;
      i -= 16;
    } while (i);
  } else {
    int i = width * height;
    assert(width == 4);
    do {
      const uint8x16_t p = vld1q_u8(pred);
      uint8x16_t r;

      r = load_unaligned_u8q(ref, ref_stride);
      ref += 4 * ref_stride;
      r = vrhaddq_u8(r, p);
      vst1q_u8(comp, r);

      pred += 16;
      comp += 16;
      i -= 16;
    } while (i);
  }
}
