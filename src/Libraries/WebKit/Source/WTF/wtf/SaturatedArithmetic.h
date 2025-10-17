/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#include <limits>
#include <stdint.h>
#include <stdlib.h>
#include <wtf/Compiler.h>

namespace WTF {

// FIXME: Enhance this so it fails to compile calls where either of the arguments can be outside the range of the integral type instead of quietly converting.
template<typename SignedIntegralType> std::enable_if_t<std::is_integral_v<SignedIntegralType> && std::is_signed_v<SignedIntegralType>, SignedIntegralType> saturatedSum(SignedIntegralType, SignedIntegralType);
template<typename UnsignedIntegralType> constexpr std::enable_if_t<std::is_integral_v<UnsignedIntegralType> && !std::is_signed_v<UnsignedIntegralType>, UnsignedIntegralType> saturatedSum(UnsignedIntegralType, UnsignedIntegralType);
template<typename IntegralType> IntegralType saturatedDifference(IntegralType, IntegralType);

inline bool signedAddInt32Overflows(int32_t a, int32_t b, int32_t& result)
{
#if COMPILER_HAS_CLANG_BUILTIN(__builtin_sadd_overflow) && !(defined __clang_major__ && __clang_major__ < 7)
    return __builtin_sadd_overflow(a, b, &result);
#else
    uint32_t ua = a;
    uint32_t ub = b;
    uint32_t uresult = ua + ub;
    result = static_cast<int32_t>(uresult);

    // Can only overflow if the signed bit of the two values match. If the signed
    // bit of the result and one of the values differ it did overflow.
    return !((ua ^ ub) >> 31) && (uresult ^ ua) >> 31;
#endif
}

template<> inline int32_t saturatedSum<int32_t>(int32_t a, int32_t b)
{
    int32_t result;
#if CPU(ARM_THUMB2)
    asm("qadd %[sum], %[addend], %[augend]"
        : [sum]"=r"(result)
        : [augend]"r"(a), [addend]"r"(b)
        : /* Nothing is clobbered. */
        );
#else
    if (signedAddInt32Overflows(a, b, result))
        result = std::numeric_limits<int32_t>::max() + (static_cast<uint32_t>(a) >> 31);
#endif
    return result;
}

inline bool signedSubtractInt32Overflows(int32_t a, int32_t b, int32_t& result)
{
#if COMPILER_HAS_CLANG_BUILTIN(__builtin_ssub_overflow) && !(defined __clang_major__ && __clang_major__ < 7)
    return __builtin_ssub_overflow(a, b, &result);
#else
    uint32_t ua = a;
    uint32_t ub = b;
    uint32_t uresult = ua - ub;
    result = static_cast<int32_t>(uresult);

    // Can only overflow if the signed bit of the two values do not match. If the
    // signed bit of the result and the first value differ it did overflow.
    return (ua ^ ub) >> 31 && (uresult ^ ua) >> 31;
#endif
}

// FIXME: Enhance this so it fails to compile calls where either of the arguments can be outside the range of int32_t instead of quietly converting.
template<> inline int32_t saturatedDifference<int32_t>(int32_t a, int32_t b)
{
    int32_t result;
#if CPU(ARM_THUMB2)
    asm("qsub %[difference], %[minuend], %[subtrahend]"
        : [difference]"=r"(result)
        : [minuend]"r"(a), [subtrahend]"r"(b)
        : /* Nothing is clobbered. */
        );
#else
    if (signedSubtractInt32Overflows(a, b, result))
        result = std::numeric_limits<int32_t>::max() + (static_cast<uint32_t>(a) >> 31);
#endif
    return result;
}

template<typename UnsignedIntegralType> constexpr std::enable_if_t<std::is_integral_v<UnsignedIntegralType> && !std::is_signed_v<UnsignedIntegralType>, UnsignedIntegralType> saturatedSum(UnsignedIntegralType a, UnsignedIntegralType b)
{
    auto sum = a + b;
    return sum < a ? std::numeric_limits<UnsignedIntegralType>::max() : sum;
}

template<typename IntegralType, typename... ArgumentTypes> constexpr uint32_t saturatedSum(IntegralType value, ArgumentTypes... arguments)
{
    return saturatedSum<IntegralType>(value, saturatedSum<IntegralType>(arguments...));
}

}

using WTF::saturatedSum;
using WTF::saturatedDifference;
