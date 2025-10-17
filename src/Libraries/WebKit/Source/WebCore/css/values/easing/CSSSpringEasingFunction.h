/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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

#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// FIXME: Contexts that allow calc() should not be defined using a closed interval - https://drafts.csswg.org/css-values-4/#calc-range
// If spring() ever goes further with standardization, the allowable ranges for `mass` and `stiffness` should be reconsidered as the
// std::nextafter() clamping is non-obvious.

// <spring()> = spring( <number [>0,âˆž]> <number [>0,âˆž]> <number [0,âˆž]> <number> )
// Non-standard
struct SpringEasingParameters {
    static constexpr auto NextAfterZero = std::numeric_limits<double>::denorm_min();
    static constexpr auto Positive = Range { NextAfterZero, Range::infinity };

    Number<Positive> mass;
    Number<Positive> stiffness;
    Number<Nonnegative> damping;
    Number<> initialVelocity;

    bool operator==(const SpringEasingParameters&) const = default;
};
using SpringEasingFunction = FunctionNotation<CSSValueSpring, SpringEasingParameters>;

template<size_t I> const auto& get(const SpringEasingParameters& value)
{
    if constexpr (!I)
        return value.mass;
    else if constexpr (I == 1)
        return value.stiffness;
    else if constexpr (I == 2)
        return value.damping;
    else if constexpr (I == 3)
        return value.initialVelocity;
}

} // namespace CSS
} // namespace WebCore

DEFINE_SPACE_SEPARATED_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::SpringEasingParameters, 4)
