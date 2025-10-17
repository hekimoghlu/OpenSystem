/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

// <xywh()> = xywh( <length-percentage>{2} <length-percentage [0,âˆž]>{2} [ round <'border-radius'> ]? )
// https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-xywh
struct Xywh {
    using Location = SpaceSeparatedPoint<LengthPercentage<>>;
    using Size = SpaceSeparatedSize<LengthPercentage<Nonnegative>>;

    Location location;
    Size size;
    BorderRadius radii;

    bool operator==(const Xywh&) const = default;
};
using XywhFunction = FunctionNotation<CSSValueXywh, Xywh>;

template<size_t I> const auto& get(const Xywh& value)
{
    if constexpr (!I)
        return value.location;
    else if constexpr (I == 1)
        return value.size;
    else if constexpr (I == 2)
        return value.radii;
}

template<> struct Serialize<Xywh> { void operator()(StringBuilder&, const Xywh&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Xywh, 3)
