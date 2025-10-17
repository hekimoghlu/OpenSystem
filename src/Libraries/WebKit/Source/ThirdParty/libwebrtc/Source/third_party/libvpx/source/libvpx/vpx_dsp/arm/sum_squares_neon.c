/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#include "vpx_dsp/arm/sum_neon.h"

uint64_t vpx_sum_squares_2d_i16_neon(const int16_t *src, int stride, int size) {
  if (size == 4) {
    int16x4_t s[4];
    int32x4_t sum_s32;

    s[0] = vld1_s16(src + 0 * stride);
    s[1] = vld1_s16(src + 1 * stride);
    s[2] = vld1_s16(src + 2 * stride);
    s[3] = vld1_s16(src + 3 * stride);

    sum_s32 = vmull_s16(s[0], s[0]);
    sum_s32 = vmlal_s16(sum_s32, s[1], s[1]);
    sum_s32 = vmlal_s16(sum_s32, s[2], s[2]);
    sum_s32 = vmlal_s16(sum_s32, s[3], s[3]);

    return horizontal_long_add_uint32x4(vreinterpretq_u32_s32(sum_s32));
  } else {
    uint64x2_t sum_u64 = vdupq_n_u64(0);
    int rows = size;

    do {
      const int16_t *src_ptr = src;
      int32x4_t sum_s32[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
      int cols = size;

      do {
        int16x8_t s[8];

        s[0] = vld1q_s16(src_ptr + 0 * stride);
        s[1] = vld1q_s16(src_ptr + 1 * stride);
        s[2] = vld1q_s16(src_ptr + 2 * stride);
        s[3] = vld1q_s16(src_ptr + 3 * stride);
        s[4] = vld1q_s16(src_ptr + 4 * stride);
        s[5] = vld1q_s16(src_ptr + 5 * stride);
        s[6] = vld1q_s16(src_ptr + 6 * stride);
        s[7] = vld1q_s16(src_ptr + 7 * stride);

        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[0]), vget_low_s16(s[0]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[1]), vget_low_s16(s[1]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[2]), vget_low_s16(s[2]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[3]), vget_low_s16(s[3]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[4]), vget_low_s16(s[4]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[5]), vget_low_s16(s[5]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[6]), vget_low_s16(s[6]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[7]), vget_low_s16(s[7]));

        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[0]), vget_high_s16(s[0]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[1]), vget_high_s16(s[1]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[2]), vget_high_s16(s[2]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[3]), vget_high_s16(s[3]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[4]), vget_high_s16(s[4]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[5]), vget_high_s16(s[5]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[6]), vget_high_s16(s[6]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[7]), vget_high_s16(s[7]));

        src_ptr += 8;
        cols -= 8;
      } while (cols);

      sum_u64 = vpadalq_u32(sum_u64, vreinterpretq_u32_s32(sum_s32[0]));
      sum_u64 = vpadalq_u32(sum_u64, vreinterpretq_u32_s32(sum_s32[1]));
      src += 8 * stride;
      rows -= 8;
    } while (rows);

    return horizontal_add_uint64x2(sum_u64);
  }
}
