/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

#include "CSSGradient.h"
#include "CSSPosition.h"
#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// <circle()> = circle( <radial-size>? [ at <position> ]? )
// https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-circle
struct Circle {
    using Extent = std::variant<Keyword::ClosestCorner, Keyword::ClosestSide, Keyword::FarthestCorner, Keyword::FarthestSide>;
    // FIXME: The spec says that this should take Length, not a LengthPercentage, but this does not match the tests.
    using Length = CSS::LengthPercentage<Nonnegative>;
    using RadialSize = std::variant<Length, Extent>;

    RadialSize radius;
    std::optional<Position> position;

    bool operator==(const Circle&) const = default;
};
using CircleFunction = FunctionNotation<CSSValueCircle, Circle>;

template<size_t I> const auto& get(const Circle& value)
{
    if constexpr (!I)
        return value.radius;
    else if constexpr (I == 1)
        return value.position;
}

template<> struct Serialize<Circle> { void operator()(StringBuilder&, const Circle&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Circle, 2)
