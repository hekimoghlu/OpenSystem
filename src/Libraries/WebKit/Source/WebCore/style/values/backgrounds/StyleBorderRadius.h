/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include "FloatRoundedRect.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

// <'border-radius'> = <length-percentage [0,âˆž]>{1,4} [ / <length-percentage [0,âˆž]>{1,4} ]?
// https://drafts.csswg.org/css-backgrounds-3/#propdef-border-radius
struct BorderRadius {
    using Corner = SpaceSeparatedSize<LengthPercentage<CSS::Nonnegative>>;

    constexpr bool operator==(const BorderRadius&) const = default;

    Corner topLeft;
    Corner topRight;
    Corner bottomRight;
    Corner bottomLeft;
};

template<size_t I> const auto& get(const BorderRadius& value)
{
    if constexpr (!I)
        return value.topLeft;
    else if constexpr (I == 1)
        return value.topRight;
    else if constexpr (I == 2)
        return value.bottomRight;
    else if constexpr (I == 3)
        return value.bottomLeft;
}

// MARK: - Conversion

template<> struct ToCSS<BorderRadius> { auto operator()(const BorderRadius&, const RenderStyle&) -> CSS::BorderRadius; };
template<> struct ToStyle<CSS::BorderRadius> { auto operator()(const CSS::BorderRadius&, const BuilderState&) -> BorderRadius; };

// MARK: - Evaluation

template<> struct Evaluation<BorderRadius> { auto operator()(const BorderRadius&, FloatSize) -> FloatRoundedRect::Radii; };

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::BorderRadius, 4)
