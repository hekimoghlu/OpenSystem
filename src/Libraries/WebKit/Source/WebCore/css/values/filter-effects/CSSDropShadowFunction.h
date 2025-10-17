/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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

#include "CSSColor.h"
#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// drop-shadow() = drop-shadow( [ <color>? && [<length>{2} <length [0,âˆž]>?] ] )
// https://drafts.fxtf.org/filter-effects/#funcdef-filter-drop-shadow
struct DropShadow {
    Markable<Color> color;
    SpaceSeparatedPoint<Length<>> location;
    Markable<Length<Nonnegative>> stdDeviation;

    bool operator==(const DropShadow&) const = default;
};
using DropShadowFunction = FunctionNotation<CSSValueDropShadow, DropShadow>;

template<size_t I> const auto& get(const DropShadow& value)
{
    if constexpr (!I)
        return value.color;
    else if constexpr (I == 1)
        return value.location;
    else if constexpr (I == 2)
        return value.stdDeviation;
}

} // namespace CSS
} // namespace WebCore

DEFINE_SPACE_SEPARATED_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::DropShadow, 3)
