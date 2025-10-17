/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
#ifndef VPX_VPX_DSP_ARM_VPX_NEON_SVE2_BRIDGE_H_
#define VPX_VPX_DSP_ARM_VPX_NEON_SVE2_BRIDGE_H_

#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_neon_sve_bridge.h>

// Some very useful instructions are exclusive to the SVE2 instruction set.
// However, we can access these instructions from a predominantly Neon context
// by making use of the Neon-SVE bridge intrinsics to reinterpret Neon vectors
// as SVE vectors - with the high part of the SVE vector (if it's longer than
// 128 bits) being "don't care".

static INLINE int16x8_t vpx_tbl2_s16(int16x8_t s0, int16x8_t s1,
                                     uint16x8_t tbl) {
  svint16x2_t samples = svcreate2_s16(svset_neonq_s16(svundef_s16(), s0),
                                      svset_neonq_s16(svundef_s16(), s1));
  return svget_neonq_s16(
      svtbl2_s16(samples, svset_neonq_u16(svundef_u16(), tbl)));
}

#endif  // VPX_VPX_DSP_ARM_VPX_NEON_SVE2_BRIDGE_H_
