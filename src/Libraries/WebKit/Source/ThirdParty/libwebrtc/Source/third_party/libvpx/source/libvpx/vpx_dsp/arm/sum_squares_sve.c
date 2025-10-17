/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_dsp/arm/vpx_neon_sve_bridge.h"

uint64_t vpx_sum_squares_2d_i16_sve(const int16_t *src, int stride, int size) {
  if (size == 4) {
    int16x4_t s[4];
    int64x2_t sum = vdupq_n_s64(0);

    s[0] = vld1_s16(src + 0 * stride);
    s[1] = vld1_s16(src + 1 * stride);
    s[2] = vld1_s16(src + 2 * stride);
    s[3] = vld1_s16(src + 3 * stride);

    int16x8_t s01 = vcombine_s16(s[0], s[1]);
    int16x8_t s23 = vcombine_s16(s[2], s[3]);

    sum = vpx_dotq_s16(sum, s01, s01);
    sum = vpx_dotq_s16(sum, s23, s23);

    return horizontal_add_uint64x2(vreinterpretq_u64_s64(sum));
  } else {
    int rows = size;
    int64x2_t sum[4] = { vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0),
                         vdupq_n_s64(0) };

    do {
      const int16_t *src_ptr = src;
      int cols = size;

      do {
        int16x8_t s[8];
        load_s16_8x8(src_ptr, stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                     &s[6], &s[7]);

        sum[0] = vpx_dotq_s16(sum[0], s[0], s[0]);
        sum[1] = vpx_dotq_s16(sum[1], s[1], s[1]);
        sum[2] = vpx_dotq_s16(sum[2], s[2], s[2]);
        sum[3] = vpx_dotq_s16(sum[3], s[3], s[3]);
        sum[0] = vpx_dotq_s16(sum[0], s[4], s[4]);
        sum[1] = vpx_dotq_s16(sum[1], s[5], s[5]);
        sum[2] = vpx_dotq_s16(sum[2], s[6], s[6]);
        sum[3] = vpx_dotq_s16(sum[3], s[7], s[7]);

        src_ptr += 8;
        cols -= 8;
      } while (cols);

      src += 8 * stride;
      rows -= 8;
    } while (rows);

    sum[0] = vaddq_s64(sum[0], sum[1]);
    sum[2] = vaddq_s64(sum[2], sum[3]);
    sum[0] = vaddq_s64(sum[0], sum[2]);

    return horizontal_add_uint64x2(vreinterpretq_u64_s64(sum[0]));
  }
}
