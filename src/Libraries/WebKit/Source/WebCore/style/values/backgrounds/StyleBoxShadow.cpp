/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "config.h"
#include "StyleBoxShadow.h"

#include "ColorBlending.h"
#include "RenderStyle.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"

namespace WebCore {
namespace Style {

// MARK: - Conversion

auto ToCSS<BoxShadow>::operator()(const BoxShadow& value, const RenderStyle& style) -> CSS::BoxShadow
{
    return {
        .color = toCSS(value.color, style),
        .location = toCSS(value.location, style),
        .blur = toCSS(value.blur, style),
        .spread = toCSS(value.spread, style),
        .inset = toCSS(value.inset, style),
        .isWebkitBoxShadow = value.isWebkitBoxShadow,
    };
}

auto ToStyle<CSS::BoxShadow>::operator()(const CSS::BoxShadow& value, const BuilderState& state) -> BoxShadow
{
    return {
        .color = value.color ? toStyle(*value.color, state) : Color::currentColor(),
        .location = toStyle(value.location, state),
        .blur = value.blur ? toStyle(*value.blur, state) : Length<CSS::Nonnegative> { 0 },
        .spread = value.spread ? toStyle(*value.spread, state) : Length<> { 0 },
        .inset = toStyle(value.inset, state),
        .isWebkitBoxShadow = value.isWebkitBoxShadow,
    };
}

// MARK: - Blending

static inline std::optional<CSS::Keyword::Inset> blendInset(std::optional<CSS::Keyword::Inset> a, std::optional<CSS::Keyword::Inset> b, const BlendingContext& context)
{
    if (a == b)
        return b;

    auto aVal = !a ? 1.0 : 0.0;
    auto bVal = !b ? 1.0 : 0.0;

    auto result = WebCore::blend(aVal, bVal, context);
    return result > 0 ? std::nullopt : std::make_optional(CSS::Keyword::Inset { });
}

auto Blending<BoxShadow>::canBlend(const BoxShadow&, const BoxShadow&, const RenderStyle&, const RenderStyle&) -> bool
{
    return true;
}

auto Blending<BoxShadow>::blend(const BoxShadow& a, const BoxShadow& b, const RenderStyle& aStyle, const RenderStyle& bStyle, const BlendingContext& context) -> BoxShadow
{
    return {
        .color = WebCore::blend(aStyle.colorResolvingCurrentColor(a.color), bStyle.colorResolvingCurrentColor(b.color), context),
        .location = WebCore::Style::blend(a.location, b.location, context),
        .blur = WebCore::Style::blend(a.blur, b.blur, context),
        .spread = WebCore::Style::blend(a.spread, b.spread, context),
        .inset = blendInset(a.inset, b.inset, context),
        .isWebkitBoxShadow = b.isWebkitBoxShadow
    };
}

} // namespace Style
} // namespace WebCore
