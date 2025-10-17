/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#include "config.h"
#include <wtf/text/StringCommon.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

SUPPRESS_ASAN
const float* findFloatAlignedImpl(const float* pointer, float target, size_t length)
{
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0b11));

    constexpr simde_uint32x4_t indexMask { 0, 1, 2, 3 };

    ASSERT(length);
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0xf));
    ASSERT((reinterpret_cast<uintptr_t>(pointer) & ~static_cast<uintptr_t>(0xf)) == reinterpret_cast<uintptr_t>(pointer));
    const float* cursor = pointer;
    constexpr size_t stride = SIMD::stride<float>;

    simde_float32x4_t targetsVector = simde_vdupq_n_f32(target);

    while (true) {
        simde_float32x4_t value = simde_vld1q_f32(cursor);
        simde_uint32x4_t mask = simde_vceqq_f32(value, targetsVector);
        if (simde_vget_lane_u64(simde_vreinterpret_u64_u16(simde_vmovn_u32(mask)), 0)) {
            simde_uint32x4_t ranked = simde_vornq_u32(indexMask, mask);
            uint32_t index = simde_vminvq_u32(ranked);
            return (index < length) ? cursor + index : nullptr;
        }
        if (length <= stride)
            return nullptr;
        length -= stride;
        cursor += stride;
    }
}

SUPPRESS_ASAN
const double* findDoubleAlignedImpl(const double* pointer, double target, size_t length)
{
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0b111));

    constexpr simde_uint32x2_t indexMask { 0, 1 };

    ASSERT(length);
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0xf));
    ASSERT((reinterpret_cast<uintptr_t>(pointer) & ~static_cast<uintptr_t>(0xf)) == reinterpret_cast<uintptr_t>(pointer));
    const double* cursor = pointer;
    constexpr size_t stride = SIMD::stride<double>;

    simde_float64x2_t targetsVector = simde_vdupq_n_f64(target);

    while (true) {
        simde_float64x2_t value = simde_vld1q_f64(cursor);
        simde_uint64x2_t mask = simde_vceqq_f64(value, targetsVector);
        simde_uint32x2_t reducedMask = simde_vmovn_u64(mask);
        if (simde_vget_lane_u64(simde_vreinterpret_u64_u32(reducedMask), 0)) {
            simde_uint32x2_t ranked = simde_vorn_u32(indexMask, reducedMask);
            uint32_t index = simde_vminv_u32(ranked);
            return (index < length) ? cursor + index : nullptr;
        }
        if (length <= stride)
            return nullptr;
        length -= stride;
        cursor += stride;
    }
}

SUPPRESS_ASAN
const LChar* find8NonASCIIAlignedImpl(std::span<const LChar> data)
{
    constexpr simde_uint8x16_t indexMask { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    auto* pointer = data.data();
    auto length = data.size();
    ASSERT(length);
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0xf));
    ASSERT((reinterpret_cast<uintptr_t>(pointer) & ~static_cast<uintptr_t>(0xf)) == reinterpret_cast<uintptr_t>(pointer));
    const uint8_t* cursor = std::bit_cast<const uint8_t*>(pointer);
    constexpr size_t stride = SIMD::stride<uint8_t>;

    simde_uint8x16_t charactersVector = simde_vdupq_n_u8(0x80);

    while (true) {
        simde_uint8x16_t value = simde_vld1q_u8(cursor);
        simde_uint8x16_t mask = simde_vcgeq_u8(value, charactersVector);
        if (simde_vmaxvq_u8(mask)) {
            simde_uint8x16_t ranked = simde_vornq_u8(indexMask, mask);
            uint8_t index = simde_vminvq_u8(ranked);
            return std::bit_cast<const LChar*>((index < length) ? cursor + index : nullptr);
        }
        if (length <= stride)
            return nullptr;
        length -= stride;
        cursor += stride;
    }
}

SUPPRESS_ASAN
const UChar* find16NonASCIIAlignedImpl(std::span<const UChar> data)
{
    auto* pointer = data.data();
    auto length = data.size();
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0x1));

    constexpr simde_uint16x8_t indexMask { 0, 1, 2, 3, 4, 5, 6, 7 };

    ASSERT(length);
    ASSERT(!(reinterpret_cast<uintptr_t>(pointer) & 0xf));
    ASSERT((reinterpret_cast<uintptr_t>(pointer) & ~static_cast<uintptr_t>(0xf)) == reinterpret_cast<uintptr_t>(pointer));
    const uint16_t* cursor = std::bit_cast<const uint16_t*>(pointer);
    constexpr size_t stride = SIMD::stride<uint16_t>;

    simde_uint16x8_t charactersVector = simde_vdupq_n_u16(0x80);

    while (true) {
        simde_uint16x8_t value = simde_vld1q_u16(cursor);
        simde_uint16x8_t mask = simde_vcgeq_u16(value, charactersVector);
        if (simde_vget_lane_u64(simde_vreinterpret_u64_u8(simde_vmovn_u16(mask)), 0)) {
            simde_uint16x8_t ranked = simde_vornq_u16(indexMask, mask);
            uint16_t index = simde_vminvq_u16(ranked);
            return std::bit_cast<const UChar*>((index < length) ? cursor + index : nullptr);
        }
        if (length <= stride)
            return nullptr;
        length -= stride;
        cursor += stride;
    }
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
