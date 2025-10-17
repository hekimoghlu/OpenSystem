/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include "CSSPolygonFunction.h"
#include "StyleFillRule.h"
#include "StylePathComputation.h"
#include "StylePrimitiveNumericTypes.h"
#include "StyleWindRuleComputation.h"

namespace WebCore {
namespace Style {

struct Polygon {
    using Vertex = SpaceSeparatedPoint<LengthPercentage<>>;
    using Vertices = CommaSeparatedVector<Vertex>;

    // FIXME: Add support the "round" clause.

    std::optional<FillRule> fillRule;
    Vertices vertices;

    bool operator==(const Polygon&) const = default;
};
using PolygonFunction = FunctionNotation<CSSValuePolygon, Polygon>;

template<size_t I> const auto& get(const Polygon& value)
{
    if constexpr (!I)
        return value.fillRule;
    else if constexpr (I == 1)
        return value.vertices;
}

DEFINE_TYPE_MAPPING(CSS::Polygon, Polygon)

template<> struct PathComputation<Polygon> { WebCore::Path operator()(const Polygon&, const FloatRect&); };
template<> struct WindRuleComputation<Polygon> { WebCore::WindRule operator()(const Polygon&); };

template<> struct Blending<Polygon> {
    auto canBlend(const Polygon&, const Polygon&) -> bool;
    auto blend(const Polygon&, const Polygon&, const BlendingContext&) -> Polygon;
};

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::Polygon, 2)
