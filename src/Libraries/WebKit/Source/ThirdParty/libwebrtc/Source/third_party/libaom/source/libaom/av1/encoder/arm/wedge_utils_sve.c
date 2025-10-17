/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

#include "aom_dsp/arm/aom_neon_sve_bridge.h"
#include "aom_dsp/arm/sum_neon.h"
#include "av1/common/reconinter.h"

uint64_t av1_wedge_sse_from_residuals_sve(const int16_t *r1, const int16_t *d,
                                          const uint8_t *m, int N) {
  assert(N % 64 == 0);

  // Predicate pattern with first 8 elements true.
  const svbool_t pattern = svptrue_pat_b16(SV_VL8);
  int64x2_t sse[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };

  int i = 0;
  do {
    int32x4_t sum[4];
    int16x8_t sum_s16[2];

    const int16x8_t r1_l = vld1q_s16(r1 + i);
    const int16x8_t r1_h = vld1q_s16(r1 + i + 8);
    const int16x8_t d_l = vld1q_s16(d + i);
    const int16x8_t d_h = vld1q_s16(d + i + 8);

    // Use a zero-extending load to widen the vector elements.
    const int16x8_t m_l = svget_neonq_s16(svld1ub_s16(pattern, m + i));
    const int16x8_t m_h = svget_neonq_s16(svld1ub_s16(pattern, m + i + 8));

    sum[0] = vshll_n_s16(vget_low_s16(r1_l), WEDGE_WEIGHT_BITS);
    sum[1] = vshll_n_s16(vget_high_s16(r1_l), WEDGE_WEIGHT_BITS);
    sum[2] = vshll_n_s16(vget_low_s16(r1_h), WEDGE_WEIGHT_BITS);
    sum[3] = vshll_n_s16(vget_high_s16(r1_h), WEDGE_WEIGHT_BITS);

    sum[0] = vmlal_s16(sum[0], vget_low_s16(m_l), vget_low_s16(d_l));
    sum[1] = vmlal_s16(sum[1], vget_high_s16(m_l), vget_high_s16(d_l));
    sum[2] = vmlal_s16(sum[2], vget_low_s16(m_h), vget_low_s16(d_h));
    sum[3] = vmlal_s16(sum[3], vget_high_s16(m_h), vget_high_s16(d_h));

    sum_s16[0] = vcombine_s16(vqmovn_s32(sum[0]), vqmovn_s32(sum[1]));
    sum_s16[1] = vcombine_s16(vqmovn_s32(sum[2]), vqmovn_s32(sum[3]));

    sse[0] = aom_sdotq_s16(sse[0], sum_s16[0], sum_s16[0]);
    sse[1] = aom_sdotq_s16(sse[1], sum_s16[1], sum_s16[1]);

    i += 16;
  } while (i < N);

  const uint64_t csse =
      (uint64_t)horizontal_add_s64x2(vaddq_s64(sse[0], sse[1]));
  return ROUND_POWER_OF_TWO(csse, 2 * WEDGE_WEIGHT_BITS);
}

int8_t av1_wedge_sign_from_residuals_sve(const int16_t *ds, const uint8_t *m,
                                         int N, int64_t limit) {
  assert(N % 16 == 0);

  // Predicate pattern with first 8 elements true.
  svbool_t pattern = svptrue_pat_b16(SV_VL8);
  int64x2_t acc_l = vdupq_n_s64(0);
  int64x2_t acc_h = vdupq_n_s64(0);

  do {
    const int16x8_t ds_l = vld1q_s16(ds);
    const int16x8_t ds_h = vld1q_s16(ds + 8);

    // Use a zero-extending load to widen the vector elements.
    const int16x8_t m_l = svget_neonq_s16(svld1ub_s16(pattern, m));
    const int16x8_t m_h = svget_neonq_s16(svld1ub_s16(pattern, m + 8));

    acc_l = aom_sdotq_s16(acc_l, ds_l, m_l);
    acc_h = aom_sdotq_s16(acc_h, ds_h, m_h);

    ds += 16;
    m += 16;
    N -= 16;
  } while (N != 0);

  const int64x2_t sum = vaddq_s64(acc_l, acc_h);
  return horizontal_add_s64x2(sum) > limit;
}
