/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

namespace WTF {

namespace dragonbox {

////////////////////////////////////////////////////////////////////////////////////////
// Return types for the main interface function.
////////////////////////////////////////////////////////////////////////////////////////

template<class UInt, bool is_signed, bool trailing_zero_flag>
struct decimal_fp;

template<class UInt>
struct decimal_fp<UInt, false, false> {
    using carrier_uint = UInt;

    carrier_uint significand;
    int32_t exponent;
};

template<class UInt>
struct decimal_fp<UInt, true, false> {
    using carrier_uint = UInt;

    carrier_uint significand;
    int32_t exponent;
    bool is_negative;
};

template<class UInt>
struct decimal_fp<UInt, false, true> {
    using carrier_uint = UInt;

    carrier_uint significand;
    int32_t exponent;
    bool may_have_trailing_zeros;
};

template<class UInt>
struct decimal_fp<UInt, true, true> {
    using carrier_uint = UInt;

    carrier_uint significand;
    int32_t exponent;
    bool may_have_trailing_zeros;
    bool is_negative;
};

template<class UInt, bool trailing_zero_flag = false>
using unsigned_decimal_fp = decimal_fp<UInt, false, trailing_zero_flag>;

template<class UInt, bool trailing_zero_flag = false>
using signed_decimal_fp = decimal_fp<UInt, true, trailing_zero_flag>;

template<class UInt>
constexpr signed_decimal_fp<UInt, false>
add_sign_to_unsigned_decimal_fp(bool is_negative, unsigned_decimal_fp<UInt, false> r) noexcept
{
    return { r.significand, r.exponent, is_negative };
}

template<class UInt>
constexpr signed_decimal_fp<UInt, true>
add_sign_to_unsigned_decimal_fp(bool is_negative, unsigned_decimal_fp<UInt, true> r) noexcept
{
    return { r.significand, r.exponent, r.may_have_trailing_zeros, is_negative };
}

namespace detail {

template<class UnsignedDecimalFp>
struct unsigned_decimal_fp_to_signed;

template<class UInt, bool trailing_zero_flag>
struct unsigned_decimal_fp_to_signed<unsigned_decimal_fp<UInt, trailing_zero_flag>> {
    using type = signed_decimal_fp<UInt, trailing_zero_flag>;
};

template<class UnsignedDecimalFp>
using unsigned_decimal_fp_to_signed_t = typename unsigned_decimal_fp_to_signed<UnsignedDecimalFp>::type;

} // namespace detail

} // namespace dragonbox

} // namespace WTF
