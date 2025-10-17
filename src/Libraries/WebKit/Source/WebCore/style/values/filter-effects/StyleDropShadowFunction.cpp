/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#include "StyleDropShadowFunction.h"

#include "CSSDropShadowFunction.h"
#include "CSSPrimitiveValue.h"
#include "Document.h"
#include "FilterOperation.h"
#include "RenderStyle.h"
#include "StyleColor.h"
#include "StylePrimitiveNumericTypes+Conversions.h"

namespace WebCore {
namespace Style {

CSS::DropShadow toCSSDropShadow(Ref<DropShadowFilterOperation> operation, const RenderStyle& style)
{
    return {
        .color = toCSS(Style::Color { operation->color() }, style),
        .location = {
            toCSS(Length<> { static_cast<float>(operation->location().x()) }, style),
            toCSS(Length<> { static_cast<float>(operation->location().y()) }, style)
        },
        .stdDeviation = toCSS(Length<CSS::Nonnegative> { static_cast<float>(operation->stdDeviation()) }, style),
    };
}

Ref<FilterOperation> createFilterOperation(const CSS::DropShadow& filter, const Document& document, RenderStyle& style, const CSSToLengthConversionData& conversionData)
{
    int x = roundForImpreciseConversion<int>(toStyle(filter.location.x(), conversionData).value);
    int y = roundForImpreciseConversion<int>(toStyle(filter.location.y(), conversionData).value);
    int stdDeviation = filter.stdDeviation ? roundForImpreciseConversion<int>(toStyle(*filter.stdDeviation, conversionData).value) : 0;
    auto color = filter.color ? style.colorResolvingCurrentColor(toStyleColorWithResolvedCurrentColor(*filter.color, document, style, conversionData, ForVisitedLink::No)) : style.color();

    return DropShadowFilterOperation::create(
        IntPoint { x, y },
        stdDeviation,
        color.isValid() ? color : WebCore::Color::transparentBlack
    );
}

} // namespace Style
} // namespace WebCore
