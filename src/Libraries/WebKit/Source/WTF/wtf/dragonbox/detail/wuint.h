/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
/*
 * License header from dragonbox
 *    https://github.com/jk-jeon/dragonbox/blob/master/LICENSE-Boost
 *    https://github.com/jk-jeon/dragonbox/blob/master/LICENSE-Apache2-LLVM
 */

#pragma once

#include <wtf/Int128.h>

namespace WTF {

namespace dragonbox {

namespace detail {

////////////////////////////////////////////////////////////////////////////////////////
// Utilities for wide unsigned integer arithmetic.
////////////////////////////////////////////////////////////////////////////////////////

namespace wuint {

inline constexpr uint64_t umul64(uint32_t x, uint32_t y) noexcept
{
    return x * static_cast<uint64_t>(y);
}

// Get 128-bit result of multiplication of two 64-bit unsigned integers.
inline constexpr UInt128 umul128(uint64_t x, uint64_t y) noexcept
{
    return static_cast<UInt128>(x) * static_cast<UInt128>(y);
}

inline constexpr uint64_t umul128_upper64(uint64_t x, uint64_t y) noexcept
{
    return static_cast<uint64_t>((static_cast<UInt128>(x) * static_cast<UInt128>(y)) >> 64);
}

// Get upper 128-bits of multiplication of a 64-bit unsigned integer and a 128-bit
// unsigned integer.
inline constexpr UInt128 umul192_upper128(uint64_t x, UInt128 y) noexcept
{
    uint64_t y_high = static_cast<uint64_t>(y >> 64);
    uint64_t y_low = static_cast<uint64_t>(y);
    auto r = umul128(x, y_high);
    r = r + static_cast<UInt128>(umul128_upper64(x, y_low));
    return r;
}

// Get upper 64-bits of multiplication of a 32-bit unsigned integer and a 64-bit
// unsigned integer.
inline constexpr uint64_t umul96_upper64(uint32_t x, uint64_t y) noexcept
{
    return umul128_upper64(static_cast<uint64_t>(x) << 32, y);
}

// Get lower 128-bits of multiplication of a 64-bit unsigned integer and a 128-bit
// unsigned integer.
inline constexpr UInt128 umul192_lower128(uint64_t x, UInt128 y) noexcept
{
    return static_cast<UInt128>(x) * y;
}

// Get lower 64-bits of multiplication of a 32-bit unsigned integer and a 64-bit
// unsigned integer.
constexpr uint64_t umul96_lower64(uint32_t x, uint64_t y) noexcept
{
    return static_cast<uint64_t>(x) * y;
}

} // namespace wuint

} // namespace detail

} // namespace dragonbox

} // namespace WTF
