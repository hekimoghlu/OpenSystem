/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include "CSSFillRule.h"
#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// <polygon()> = polygon( <'fill-rule'>? [ round <length> ]? , [<length-percentage> <length-percentage>]# )
// https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-polygon
// FIXME: Add support the "round" clause.
struct Polygon {
    using Vertex = SpaceSeparatedPoint<LengthPercentage<>>;
    using Vertices = CommaSeparatedVector<Vertex>;

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

template<> struct Serialize<Polygon> { void operator()(StringBuilder&, const Polygon&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Polygon, 2)
