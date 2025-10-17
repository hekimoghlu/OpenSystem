/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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

#include "aom_dsp/txfm_common.h"
#include "config/av1_rtcd.h"

static void transpose4x4(int16x8_t in[2], int16x4_t out[4]) {
  int32x4x2_t b0 =
      vtrnq_s32(vreinterpretq_s32_s16(in[0]), vreinterpretq_s32_s16(in[1]));
  int16x4x2_t c0 = vtrn_s16(vreinterpret_s16_s32(vget_low_s32(b0.val[0])),
                            vreinterpret_s16_s32(vget_high_s32(b0.val[0])));
  int16x4x2_t c1 = vtrn_s16(vreinterpret_s16_s32(vget_low_s32(b0.val[1])),
                            vreinterpret_s16_s32(vget_high_s32(b0.val[1])));
  out[0] = c0.val[0];
  out[1] = c0.val[1];
  out[2] = c1.val[0];
  out[3] = c1.val[1];
}

void av1_fwht4x4_neon(const int16_t *input, tran_low_t *output, int stride) {
  // Load the 4x4 source in transposed form.
  int16x4_t a1, b1, c1, d1, e;
  a1 = vld1_s16(&input[0]);
  b1 = vld1_s16(&input[1 * stride]);
  c1 = vld1_s16(&input[2 * stride]);
  d1 = vld1_s16(&input[3 * stride]);

  // WHT.

  // Row transforms.
  a1 = vadd_s16(a1, b1);
  d1 = vsub_s16(d1, c1);
  e = vhsub_s16(a1, d1);
  b1 = vsub_s16(e, b1);
  c1 = vsub_s16(e, c1);
  a1 = vsub_s16(a1, c1);
  d1 = vadd_s16(d1, b1);

  int16x8_t x[2];
  x[0] = vcombine_s16(a1, c1);
  x[1] = vcombine_s16(d1, b1);

  int16x4_t s[4];
  transpose4x4(x, s);

  a1 = s[0];
  b1 = s[1];
  c1 = s[2];
  d1 = s[3];

  // Row transforms.
  a1 = vadd_s16(a1, b1);
  d1 = vsub_s16(d1, c1);
  e = vhsub_s16(a1, d1);
  b1 = vsub_s16(e, b1);
  c1 = vsub_s16(e, c1);
  a1 = vsub_s16(a1, c1);
  d1 = vadd_s16(d1, b1);

  vst1q_s32(&output[0], vshll_n_s16(a1, UNIT_QUANT_SHIFT));
  vst1q_s32(&output[4], vshll_n_s16(c1, UNIT_QUANT_SHIFT));
  vst1q_s32(&output[8], vshll_n_s16(d1, UNIT_QUANT_SHIFT));
  vst1q_s32(&output[12], vshll_n_s16(b1, UNIT_QUANT_SHIFT));
}
