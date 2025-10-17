/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include <assert.h>
#include <arm_neon.h>

#include "config/av1_rtcd.h"
#include "aom_dsp/arm/sum_neon.h"

int64_t av1_highbd_block_error_neon(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff,
                                    intptr_t block_size, int64_t *ssz, int bd) {
  uint64x2_t err_u64 = vdupq_n_u64(0);
  int64x2_t ssz_s64 = vdupq_n_s64(0);

  const int shift = 2 * (bd - 8);
  const int rounding = shift > 0 ? 1 << (shift - 1) : 0;

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int32x4_t c = vld1q_s32(coeff);
    const int32x4_t d = vld1q_s32(dqcoeff);

    const uint32x4_t diff = vreinterpretq_u32_s32(vabdq_s32(c, d));

    err_u64 = vmlal_u32(err_u64, vget_low_u32(diff), vget_low_u32(diff));
    err_u64 = vmlal_u32(err_u64, vget_high_u32(diff), vget_high_u32(diff));

    ssz_s64 = vmlal_s32(ssz_s64, vget_low_s32(c), vget_low_s32(c));
    ssz_s64 = vmlal_s32(ssz_s64, vget_high_s32(c), vget_high_s32(c));

    coeff += 4;
    dqcoeff += 4;
    block_size -= 4;
  } while (block_size != 0);

  *ssz = (horizontal_add_s64x2(ssz_s64) + rounding) >> shift;
  return ((int64_t)horizontal_add_u64x2(err_u64) + rounding) >> shift;
}
