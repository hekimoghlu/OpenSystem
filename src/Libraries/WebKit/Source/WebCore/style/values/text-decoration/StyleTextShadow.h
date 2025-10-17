/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

#include "CSSTextShadow.h"
#include "StyleColor.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

struct TextShadow {
    Color color;
    SpaceSeparatedPoint<Length<>> location;
    Length<CSS::Nonnegative> blur;

    bool operator==(const TextShadow&) const = default;
};

template<size_t I> const auto& get(const TextShadow& value)
{
    if constexpr (!I)
        return value.color;
    else if constexpr (I == 1)
        return value.location;
    else if constexpr (I == 2)
        return value.blur;
}

template<> struct ToCSS<TextShadow> { auto operator()(const TextShadow&, const RenderStyle&) -> CSS::TextShadow; };
template<> struct ToStyle<CSS::TextShadow> { auto operator()(const CSS::TextShadow&, const BuilderState&) -> TextShadow; };

template<> struct Blending<TextShadow> {
    auto canBlend(const TextShadow&, const TextShadow&, const RenderStyle&, const RenderStyle&) -> bool;
    auto blend(const TextShadow&, const TextShadow&, const RenderStyle&, const RenderStyle&, const BlendingContext&) -> TextShadow;
};

} // namespace Style
} // namespace WebCore

DEFINE_SPACE_SEPARATED_TUPLE_LIKE_CONFORMANCE(WebCore::Style::TextShadow, 3)
