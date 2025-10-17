/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

#include "CSSEllipseFunction.h"
#include "StylePathComputation.h"
#include "StylePosition.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

struct Ellipse {
    using Extent = CSS::Ellipse::Extent;
    using Length = Style::LengthPercentage<CSS::Nonnegative>;
    using RadialSize = std::variant<Length, Extent>;

    SpaceSeparatedPair<RadialSize> radii;
    std::optional<Position> position;

    bool operator==(const Ellipse&) const = default;
};
using EllipseFunction = FunctionNotation<CSSValueEllipse, Ellipse>;

template<size_t I> const auto& get(const Ellipse& value)
{
    if constexpr (!I)
        return value.radii;
    else if constexpr (I == 1)
        return value.position;
}

DEFINE_TYPE_MAPPING(CSS::Ellipse, Ellipse)

FloatPoint resolvePosition(const Ellipse& value, FloatSize boxSize);
FloatSize resolveRadii(const Ellipse&, FloatSize boxSize, FloatPoint center);
WebCore::Path pathForCenterCoordinate(const Ellipse&, const FloatRect&, FloatPoint);

template<> struct PathComputation<Ellipse> { WebCore::Path operator()(const Ellipse&, const FloatRect&); };

template<> struct Blending<Ellipse> {
    auto canBlend(const Ellipse&, const Ellipse&) -> bool;
    auto blend(const Ellipse&, const Ellipse&, const BlendingContext&) -> Ellipse;
};

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::Ellipse, 2)
