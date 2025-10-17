/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include <algorithm>
#include <wtf/dtoa/utils.h>

namespace WTF {

namespace dragonbox {

namespace detail {

////////////////////////////////////////////////////////////////////////////////////////
// Some basic features for encoding/decoding IEEE-754 formats.
////////////////////////////////////////////////////////////////////////////////////////

template<class T>
typename std::add_rvalue_reference<T>::type declval() noexcept;

template<class T>
struct physical_bits {
    static constexpr size_t value = sizeof(T) * std::numeric_limits<unsigned char>::digits;
};

template<class T>
struct value_bits {
    static constexpr size_t value = std::numeric_limits<typename std::enable_if<std::is_unsigned<T>::value, T>::type>::digits;
};

////////////////////////////////////////////////////////////////////////////////////////
// Some simple utilities for constexpr computation.
////////////////////////////////////////////////////////////////////////////////////////

template <int32_t k, class Int>
constexpr Int compute_power(Int a) noexcept {
    static_assert(k >= 0, "");
    Int p = 1;
    for (int32_t i = 0; i < k; ++i)
        p *= a;
    return p;
}

template<int32_t a, class UInt>
constexpr int32_t count_factors(UInt n) noexcept
{
    static_assert(a > 1, "");
    int32_t c = 0;
    while (!(n % a)) {
        n /= a;
        ++c;
    }
    return c;
}

} // namespace detail

} // namespace dragonbox

} // namespace WTF
