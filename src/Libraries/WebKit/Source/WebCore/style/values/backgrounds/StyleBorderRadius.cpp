/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "StyleBorderRadius.h"

#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"

namespace WebCore {
namespace Style {

auto ToCSS<BorderRadius>::operator()(const BorderRadius& value, const RenderStyle& style) -> CSS::BorderRadius
{
    return {
        .horizontal {
            toCSS(value.topLeft.width(), style),
            toCSS(value.topRight.width(), style),
            toCSS(value.bottomRight.width(), style),
            toCSS(value.bottomLeft.width(), style),
        },
        .vertical {
            toCSS(value.topLeft.height(), style),
            toCSS(value.topRight.height(), style),
            toCSS(value.bottomRight.height(), style),
            toCSS(value.bottomLeft.height(), style),
        },
    };
}

auto ToStyle<CSS::BorderRadius>::operator()(const CSS::BorderRadius& value, const BuilderState& state) -> BorderRadius
{
    return {
        .topLeft { toStyle(value.topLeft(), state) },
        .topRight { toStyle(value.topRight(), state) },
        .bottomRight { toStyle(value.bottomRight(), state) },
        .bottomLeft { toStyle(value.bottomLeft(), state) },
    };
}

auto Evaluation<BorderRadius>::operator()(const BorderRadius& value, FloatSize referenceBox) -> FloatRoundedRect::Radii
{
    return {
        evaluate(value.topLeft, referenceBox),
        evaluate(value.topRight, referenceBox),
        evaluate(value.bottomLeft, referenceBox),
        evaluate(value.bottomRight, referenceBox)
    };
}

} // namespace Style
} // namespace WebCore
