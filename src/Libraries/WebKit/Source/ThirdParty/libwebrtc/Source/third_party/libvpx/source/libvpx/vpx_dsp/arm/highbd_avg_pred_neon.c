/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include "./vpx_config.h"

void vpx_highbd_comp_avg_pred_neon(uint16_t *comp_pred, const uint16_t *pred,
                                   int width, int height, const uint16_t *ref,
                                   int ref_stride) {
  int i = height;
  if (width > 8) {
    do {
      int j = 0;
      do {
        const uint16x8_t p = vld1q_u16(pred + j);
        const uint16x8_t r = vld1q_u16(ref + j);

        uint16x8_t avg = vrhaddq_u16(p, r);
        vst1q_u16(comp_pred + j, avg);

        j += 8;
      } while (j < width);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  } else if (width == 8) {
    do {
      const uint16x8_t p = vld1q_u16(pred);
      const uint16x8_t r = vld1q_u16(ref);

      uint16x8_t avg = vrhaddq_u16(p, r);
      vst1q_u16(comp_pred, avg);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  } else {
    assert(width == 4);
    do {
      const uint16x4_t p = vld1_u16(pred);
      const uint16x4_t r = vld1_u16(ref);

      uint16x4_t avg = vrhadd_u16(p, r);
      vst1_u16(comp_pred, avg);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  }
}
