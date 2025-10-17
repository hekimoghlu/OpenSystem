/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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

#include "CSSBorderRadius.h"
#include "CSSPrimitiveNumericTypes.h"
#include "RectEdges.h"

namespace WebCore {
namespace CSS {

// <rect()> = rect( [ <length-percentage> | auto ]{4} [ round <'border-radius'> ]? )
// https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-rect
struct Rect {
    using Edge = std::variant<LengthPercentage<>, Keyword::Auto>;

    SpaceSeparatedRectEdges<Edge> edges;
    BorderRadius radii;

    bool operator==(const Rect&) const = default;
};
using RectFunction = FunctionNotation<CSSValueRect, Rect>;

template<size_t I> const auto& get(const Rect& value)
{
    if constexpr (!I)
        return value.edges;
    else if constexpr (I == 1)
        return value.radii;
}

template<> struct Serialize<Rect> { void operator()(StringBuilder&, const Rect&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Rect, 2)
