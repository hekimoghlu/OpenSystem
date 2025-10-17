/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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

// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BASE_NUMERICS_SAFE_CONVERSIONS_ARM_IMPL_H_
#define BASE_NUMERICS_SAFE_CONVERSIONS_ARM_IMPL_H_

#include <cassert>
#include <limits>
#include <type_traits>

#include "anglebase/numerics/safe_conversions_impl.h"

namespace angle
{
namespace base
{
namespace internal
{

// Fast saturation to a destination type.
template <typename Dst, typename Src>
struct SaturateFastAsmOp
{
    static constexpr bool is_supported =
        std::is_signed<Src>::value && std::is_integral<Dst>::value &&
        std::is_integral<Src>::value &&
        IntegerBitsPlusSign<Src>::value <= IntegerBitsPlusSign<int32_t>::value &&
        IntegerBitsPlusSign<Dst>::value <= IntegerBitsPlusSign<int32_t>::value &&
        !IsTypeInRangeForNumericType<Dst, Src>::value;

    __attribute__((always_inline)) static Dst Do(Src value)
    {
        int32_t src = value;
        typename std::conditional<std::is_signed<Dst>::value, int32_t, uint32_t>::type result;
        if (std::is_signed<Dst>::value)
        {
            asm("ssat %[dst], %[shift], %[src]"
                : [dst] "=r"(result)
                : [src] "r"(src), [shift] "n"(IntegerBitsPlusSign<Dst>::value <= 32
                                                  ? IntegerBitsPlusSign<Dst>::value
                                                  : 32));
        }
        else
        {
            asm("usat %[dst], %[shift], %[src]"
                : [dst] "=r"(result)
                : [src] "r"(src), [shift] "n"(IntegerBitsPlusSign<Dst>::value < 32
                                                  ? IntegerBitsPlusSign<Dst>::value
                                                  : 31));
        }
        return static_cast<Dst>(result);
    }
};

}  // namespace internal
}  // namespace base
}  // namespace angle

#endif  // BASE_NUMERICS_SAFE_CONVERSIONS_ARM_IMPL_H_
