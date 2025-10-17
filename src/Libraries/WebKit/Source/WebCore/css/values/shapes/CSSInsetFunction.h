/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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

namespace WebCore {
namespace CSS {

// <inset()> = inset( <length-percentage>{1,4} [ round <'border-radius'> ]? )
// https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-inset
struct Inset {
    using Insets = MinimallySerializingSpaceSeparatedRectEdges<LengthPercentage<>>;

    Insets insets;
    BorderRadius radii;

    bool operator==(const Inset&) const = default;
};
using InsetFunction = FunctionNotation<CSSValueInset, Inset>;

template<size_t I> const auto& get(const Inset& value)
{
    if constexpr (!I)
        return value.insets;
    else if constexpr (I == 1)
        return value.radii;
}

template<> struct Serialize<Inset> { void operator()(StringBuilder&, const Inset&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Inset, 2)
