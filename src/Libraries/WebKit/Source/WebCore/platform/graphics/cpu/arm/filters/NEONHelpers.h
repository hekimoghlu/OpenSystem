/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#pragma once

#if HAVE(ARM_NEON_INTRINSICS)

#include <arm_neon.h>

namespace WebCore {

inline float32x4_t loadRGBA8AsFloat(const uint32_t* source)
{
    uint32x2_t temporary1 = {0, 0};
    temporary1 = vset_lane_u32(*source, temporary1, 0);
    uint16x4_t temporary2 = vget_low_u16(vmovl_u8(vreinterpret_u8_u32(temporary1)));
    return vcvtq_f32_u32(vmovl_u16(temporary2));
}

inline void storeFloatAsRGBA8(float32x4_t data, uint32_t* destination)
{
    uint16x4_t temporary1 = vmovn_u32(vcvtq_u32_f32(data));
    uint8x8_t temporary2 = vmovn_u16(vcombine_u16(temporary1, temporary1));
    *destination = vget_lane_u32(vreinterpret_u32_u8(temporary2), 0);
}

} // namespace WebCore

#endif // HAVE(ARM_NEON_INTRINSICS)
