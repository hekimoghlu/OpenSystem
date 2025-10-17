/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#ifndef VPX_VPX_DSP_ARM_VPX_NEON_SVE_BRIDGE_H_
#define VPX_VPX_DSP_ARM_VPX_NEON_SVE_BRIDGE_H_

#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_neon_sve_bridge.h>

// Dot product instructions operating on 16-bit input elements are exclusive to
// the SVE instruction set. However, we can access these instructions from a
// predominantly Neon context by making use of the Neon-SVE bridge intrinsics
// to reinterpret Neon vectors as SVE vectors - with the high part of the SVE
// vector (if it's longer than 128 bits) being "don't care".

// While sub-optimal on machines that have SVE vector length > 128-bit - as the
// remainder of the vector is unused - this approach is still beneficial when
// compared to a Neon-only solution.

static INLINE uint64x2_t vpx_dotq_u16(uint64x2_t acc, uint16x8_t x,
                                      uint16x8_t y) {
  return svget_neonq_u64(svdot_u64(svset_neonq_u64(svundef_u64(), acc),
                                   svset_neonq_u16(svundef_u16(), x),
                                   svset_neonq_u16(svundef_u16(), y)));
}

static INLINE int64x2_t vpx_dotq_s16(int64x2_t acc, int16x8_t x, int16x8_t y) {
  return svget_neonq_s64(svdot_s64(svset_neonq_s64(svundef_s64(), acc),
                                   svset_neonq_s16(svundef_s16(), x),
                                   svset_neonq_s16(svundef_s16(), y)));
}

#define vpx_dotq_lane_s16(acc, x, y, lane)                            \
  svget_neonq_s64(svdot_lane_s64(svset_neonq_s64(svundef_s64(), acc), \
                                 svset_neonq_s16(svundef_s16(), x),   \
                                 svset_neonq_s16(svundef_s16(), y), lane))

static INLINE uint16x8_t vpx_tbl_u16(uint16x8_t data, uint16x8_t indices) {
  return svget_neonq_u16(svtbl_u16(svset_neonq_u16(svundef_u16(), data),
                                   svset_neonq_u16(svundef_u16(), indices)));
}

#endif  // VPX_VPX_DSP_ARM_VPX_NEON_SVE_BRIDGE_H_
