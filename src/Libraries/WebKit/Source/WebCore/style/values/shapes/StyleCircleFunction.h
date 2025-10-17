/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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

#include "CSSCircleFunction.h"
#include "StylePathComputation.h"
#include "StylePosition.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

struct Circle {
    using Extent = CSS::Circle::Extent;
    using Length = Style::LengthPercentage<CSS::Nonnegative>;
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

DEFINE_TYPE_MAPPING(CSS::Circle, Circle)

FloatPoint resolvePosition(const Circle& value, FloatSize boxSize);
float resolveRadius(const Circle& value, FloatSize boxSize, FloatPoint center);
WebCore::Path pathForCenterCoordinate(const Circle&, const FloatRect&, FloatPoint);

template<> struct PathComputation<Circle> { WebCore::Path operator()(const Circle&, const FloatRect&); };

template<> struct Blending<Circle> {
    auto canBlend(const Circle&, const Circle&) -> bool;
    auto blend(const Circle&, const Circle&, const BlendingContext&) -> Circle;
};

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::Circle, 2)
