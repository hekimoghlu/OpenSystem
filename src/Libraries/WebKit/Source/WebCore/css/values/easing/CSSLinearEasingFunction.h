/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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

// <linear()> = linear( [ <number> && <percentage>{0,2} ]# )
// https://drafts.csswg.org/css-easing-2/#funcdef-linear
struct LinearEasingParameters {
    struct Stop {
        struct Length {
            Percentage<> input;
            std::optional<Percentage<>> extra;

            bool operator==(const Length&) const = default;
        };

        Number<> output;
        std::optional<Length> input;

        bool operator==(const Stop&) const = default;
    };

    CommaSeparatedVector<Stop> stops;

    bool operator==(const LinearEasingParameters&) const = default;
};
using LinearEasingFunction = FunctionNotation<CSSValueLinear, LinearEasingParameters>;

DEFINE_TYPE_WRAPPER_GET(LinearEasingParameters, stops);

template<size_t I> const auto& get(const LinearEasingParameters::Stop& value)
{
    if constexpr (!I)
        return value.output;
    else if constexpr (I == 1)
        return value.input;
}

template<size_t I> const auto& get(const LinearEasingParameters::Stop::Length& value)
{
    if constexpr (!I)
        return value.input;
    else if constexpr (I == 1)
        return value.extra;
}

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::LinearEasingParameters, 1)
DEFINE_SPACE_SEPARATED_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::LinearEasingParameters::Stop, 2)
DEFINE_SPACE_SEPARATED_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::LinearEasingParameters::Stop::Length, 2)
