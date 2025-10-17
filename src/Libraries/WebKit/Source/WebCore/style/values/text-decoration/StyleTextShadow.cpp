/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "StyleTextShadow.h"

#include "ColorBlending.h"
#include "RenderStyle.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"

namespace WebCore {
namespace Style {

// MARK: - Conversion

auto ToCSS<TextShadow>::operator()(const TextShadow& value, const RenderStyle& style) -> CSS::TextShadow
{
    return {
        .color = toCSS(value.color, style),
        .location = toCSS(value.location, style),
        .blur = toCSS(value.blur, style),
    };
}

auto ToStyle<CSS::TextShadow>::operator()(const CSS::TextShadow& value, const BuilderState& state) -> TextShadow
{
    return {
        .color = value.color ? toStyle(*value.color, state) : Color::currentColor(),
        .location = toStyle(value.location, state),
        .blur = value.blur ? toStyle(*value.blur, state) : Length<CSS::Nonnegative> { 0 },
    };
}

// MARK: - Blending

auto Blending<TextShadow>::canBlend(const TextShadow&, const TextShadow&, const RenderStyle&, const RenderStyle&) -> bool
{
    return true;
}

auto Blending<TextShadow>::blend(const TextShadow& a, const TextShadow& b, const RenderStyle& aStyle, const RenderStyle& bStyle, const BlendingContext& context) -> TextShadow
{
    return {
        .color = WebCore::blend(aStyle.colorResolvingCurrentColor(a.color), bStyle.colorResolvingCurrentColor(b.color), context),
        .location = WebCore::Style::blend(a.location, b.location, context),
        .blur = WebCore::Style::blend(a.blur, b.blur, context),
    };
}

} // namespace Style
} // namespace WebCore
