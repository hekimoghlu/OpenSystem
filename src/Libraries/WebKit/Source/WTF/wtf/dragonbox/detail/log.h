/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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

#include <wtf/dragonbox/detail/util.h>

namespace WTF {

namespace dragonbox {

namespace detail {

////////////////////////////////////////////////////////////////////////////////////////
// Utilities for fast/constexpr log computation.
////////////////////////////////////////////////////////////////////////////////////////

namespace log {

static_assert((-1 >> 1) == -1, "right-shift for signed integers must be arithmetic");

// Compute floor(e * c - s).
enum class multiply : uint32_t { };
enum class subtract : uint32_t { };
enum class shift : size_t { };
enum class min_exponent : int32_t { };
enum class max_exponent : int32_t { };

template<multiply m, subtract f, shift k, min_exponent e_min, max_exponent e_max>
constexpr int32_t compute(int32_t e) noexcept
{
    ASSERT(static_cast<int32_t>(e_min) <= e && e <= static_cast<int32_t>(e_max));
    return static_cast<int32_t>((static_cast<int32_t>(e) * static_cast<int32_t>(m) - static_cast<int32_t>(f)) >> size_t(k));
}

// For constexpr computation.
// Returns -1 when n = 0.
template<class UInt>
constexpr int32_t floor_log2(UInt n) noexcept
{
    int32_t count = -1;
    while (n) {
        ++count;
        n >>= 1;
    }
    return count;
}

static constexpr int32_t floor_log10_pow2_min_exponent = -2620;
static constexpr int32_t floor_log10_pow2_max_exponent = 2620;
constexpr int32_t floor_log10_pow2(int32_t e) noexcept
{
    return compute<multiply(315653), subtract(0), shift(20), min_exponent(floor_log10_pow2_min_exponent), max_exponent(floor_log10_pow2_max_exponent)>(e);
}

static constexpr int32_t floor_log2_pow10_min_exponent = -1233;
static constexpr int32_t floor_log2_pow10_max_exponent = 1233;
constexpr int32_t floor_log2_pow10(int32_t e) noexcept
{
    return compute<multiply(1741647), subtract(0), shift(19), min_exponent(floor_log2_pow10_min_exponent), max_exponent(floor_log2_pow10_max_exponent)>(e);
}

static constexpr int32_t floor_log10_pow2_minus_log10_4_over_3_min_exponent = -2985;
static constexpr int32_t floor_log10_pow2_minus_log10_4_over_3_max_exponent = 2936;
constexpr int32_t floor_log10_pow2_minus_log10_4_over_3(int32_t e) noexcept
{
    return compute<multiply(631305), subtract(261663), shift(21), min_exponent(floor_log10_pow2_minus_log10_4_over_3_min_exponent), max_exponent(floor_log10_pow2_minus_log10_4_over_3_max_exponent)>(e);
}

static constexpr int32_t floor_log5_pow2_min_exponent = -1831;
static constexpr int32_t floor_log5_pow2_max_exponent = 1831;
constexpr int32_t floor_log5_pow2(int32_t e) noexcept
{
    return compute<multiply(225799), subtract(0), shift(19), min_exponent(floor_log5_pow2_min_exponent), max_exponent(floor_log5_pow2_max_exponent)>(e);
}

static constexpr int32_t floor_log5_pow2_minus_log5_3_min_exponent = -3543;
static constexpr int32_t floor_log5_pow2_minus_log5_3_max_exponent = 2427;
constexpr int32_t floor_log5_pow2_minus_log5_3(int32_t e) noexcept
{
    return compute<multiply(451597), subtract(715764), shift(20), min_exponent(floor_log5_pow2_minus_log5_3_min_exponent), max_exponent(floor_log5_pow2_minus_log5_3_max_exponent)>(e);
}

} // namespace log

} // namespace detail

} // namespace dragonbox

} // namespace WTF
