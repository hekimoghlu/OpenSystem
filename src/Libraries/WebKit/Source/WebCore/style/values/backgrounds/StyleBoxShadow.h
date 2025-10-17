/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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

#include "CSSBoxShadow.h"
#include "StyleColor.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

struct BoxShadow {
    Color color;
    SpaceSeparatedPoint<Length<>> location;
    Length<CSS::Nonnegative> blur;
    Length<> spread;
    std::optional<CSS::Keyword::Inset> inset;
    bool isWebkitBoxShadow;

    bool operator==(const BoxShadow&) const = default;
};

template<size_t I> const auto& get(const BoxShadow& value)
{
    if constexpr (!I)
        return value.color;
    else if constexpr (I == 1)
        return value.location;
    else if constexpr (I == 2)
        return value.blur;
    else if constexpr (I == 3)
        return value.spread;
    else if constexpr (I == 4)
        return value.inset;
}

template<> struct ToCSS<BoxShadow> { auto operator()(const BoxShadow&, const RenderStyle&) -> CSS::BoxShadow; };
template<> struct ToStyle<CSS::BoxShadow> { auto operator()(const CSS::BoxShadow&, const BuilderState&) -> BoxShadow; };

template<> struct Blending<BoxShadow> {
    auto canBlend(const BoxShadow&, const BoxShadow&, const RenderStyle&, const RenderStyle&) -> bool;
    auto blend(const BoxShadow&, const BoxShadow&, const RenderStyle&, const RenderStyle&, const BlendingContext&) -> BoxShadow;
};

} // namespace Style
} // namespace WebCore

DEFINE_SPACE_SEPARATED_TUPLE_LIKE_CONFORMANCE(WebCore::Style::BoxShadow, 5)
