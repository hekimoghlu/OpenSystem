/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#ifndef VPX_VPX_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_
#define VPX_VPX_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_

#include <arm_neon.h>

static INLINE uint16x4_t highbd_convolve4_4_neon(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t filters, const uint16x4_t max) {
  int32x4_t sum = vmull_lane_s16(s0, filters, 0);
  sum = vmlal_lane_s16(sum, s1, filters, 1);
  sum = vmlal_lane_s16(sum, s2, filters, 2);
  sum = vmlal_lane_s16(sum, s3, filters, 3);

  uint16x4_t res = vqrshrun_n_s32(sum, FILTER_BITS);
  return vmin_u16(res, max);
}

static INLINE uint16x8_t highbd_convolve4_8_neon(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x4_t filters, const uint16x8_t max) {
  int32x4_t sum0 = vmull_lane_s16(vget_low_s16(s0), filters, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s1), filters, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s2), filters, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s3), filters, 3);

  int32x4_t sum1 = vmull_lane_s16(vget_high_s16(s0), filters, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s1), filters, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s2), filters, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s3), filters, 3);

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(sum0, FILTER_BITS),
                                vqrshrun_n_s32(sum1, FILTER_BITS));
  return vminq_u16(res, max);
}

#endif  // VPX_VPX_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_
