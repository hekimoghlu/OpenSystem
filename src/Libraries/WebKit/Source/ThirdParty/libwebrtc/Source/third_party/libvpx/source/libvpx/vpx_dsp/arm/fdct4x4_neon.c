/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/fdct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/fdct4x4_neon.h"

void vpx_fdct4x4_neon(const int16_t *input, tran_low_t *final_output,
                      int stride) {
  // input[M * stride] * 16
  int16x4_t in[4];
  in[0] = vshl_n_s16(vld1_s16(input + 0 * stride), 4);
  in[1] = vshl_n_s16(vld1_s16(input + 1 * stride), 4);
  in[2] = vshl_n_s16(vld1_s16(input + 2 * stride), 4);
  in[3] = vshl_n_s16(vld1_s16(input + 3 * stride), 4);

  // If the very first value != 0, then add 1.
  if (input[0] != 0) {
    const int16x4_t one = vreinterpret_s16_s64(vdup_n_s64(1));
    in[0] = vadd_s16(in[0], one);
  }
  vpx_fdct4x4_pass1_neon(in);
  vpx_fdct4x4_pass2_neon(in);
  {
    // Not quite a rounding shift. Only add 1 despite shifting by 2.
    const int16x8_t one = vdupq_n_s16(1);
    int16x8_t out_01 = vcombine_s16(in[0], in[1]);
    int16x8_t out_23 = vcombine_s16(in[2], in[3]);
    out_01 = vshrq_n_s16(vaddq_s16(out_01, one), 2);
    out_23 = vshrq_n_s16(vaddq_s16(out_23, one), 2);
    store_s16q_to_tran_low(final_output + 0 * 8, out_01);
    store_s16q_to_tran_low(final_output + 1 * 8, out_23);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH

void vpx_highbd_fdct4x4_neon(const int16_t *input, tran_low_t *final_output,
                             int stride) {
  const int32x4_t const_one = vdupq_n_s32(1);

  // input[M * stride] * 16
  int32x4_t in[4];
  in[0] = vshll_n_s16(vld1_s16(input + 0 * stride), 4);
  in[1] = vshll_n_s16(vld1_s16(input + 1 * stride), 4);
  in[2] = vshll_n_s16(vld1_s16(input + 2 * stride), 4);
  in[3] = vshll_n_s16(vld1_s16(input + 3 * stride), 4);

  // If the very first value != 0, then add 1.
  if (input[0] != 0) {
    static const int32_t k1000[4] = { 1, 0, 0, 0 };
    in[0] = vaddq_s32(in[0], vld1q_s32(k1000));
  }

  vpx_highbd_fdct4x4_pass1_neon(in);
  vpx_highbd_fdct4x4_pass1_neon(in);
  {
    // Not quite a rounding shift. Only add 1 despite shifting by 2.
    in[0] = vshrq_n_s32(vaddq_s32(in[0], const_one), 2);
    in[1] = vshrq_n_s32(vaddq_s32(in[1], const_one), 2);
    in[2] = vshrq_n_s32(vaddq_s32(in[2], const_one), 2);
    in[3] = vshrq_n_s32(vaddq_s32(in[3], const_one), 2);

    vst1q_s32(final_output, in[0]);
    vst1q_s32(final_output + 4, in[1]);
    vst1q_s32(final_output + 8, in[2]);
    vst1q_s32(final_output + 12, in[3]);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
