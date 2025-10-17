/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include "StyleBlurFunction.h"

#include "CSSBlurFunction.h"
#include "CSSFilterFunctionDescriptor.h"
#include "FilterOperation.h"
#include "StylePrimitiveNumericTypes+Conversions.h"

namespace WebCore {
namespace Style {

CSS::Blur toCSSBlur(Ref<BlurFilterOperation> operation, const RenderStyle& style)
{
    return { CSS::Blur::Parameter { toCSS(Length<CSS::Nonnegative> { operation->stdDeviation().value() }, style) } };
}

Ref<FilterOperation> createFilterOperation(const CSS::Blur& filter, const Document&, RenderStyle&, const CSSToLengthConversionData& conversionData)
{
    WebCore::Length stdDeviation;
    if (auto parameter = filter.value)
        stdDeviation = WebCore::Length { toStyle(*parameter, conversionData).value, LengthType::Fixed };
    else
        stdDeviation = WebCore::Length { filterFunctionDefaultValue<CSS::BlurFunction::name>().value, LengthType::Fixed };

    return BlurFilterOperation::create(WTFMove(stdDeviation));
}

} // namespace Style
} // namespace WebCore
