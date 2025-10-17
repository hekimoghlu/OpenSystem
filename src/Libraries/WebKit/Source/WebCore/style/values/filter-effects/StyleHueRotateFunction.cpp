/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#include "StyleHueRotateFunction.h"

#include "CSSFilterFunctionDescriptor.h"
#include "CSSHueRotateFunction.h"
#include "FilterOperation.h"
#include "StylePrimitiveNumericTypes+Conversions.h"

namespace WebCore {
namespace Style {

CSS::HueRotate toCSSHueRotate(Ref<BasicColorMatrixFilterOperation> operation, const RenderStyle& style)
{
    return { CSS::HueRotate::Parameter { toCSS(Angle<> { operation->amount() }, style) } };
}

Ref<FilterOperation> createFilterOperation(const CSS::HueRotate& filter, const Document&, RenderStyle&, const CSSToLengthConversionData& conversionData)
{
    double value;
    if (auto parameter = filter.value)
        value = toStyle(*parameter, conversionData).value;
    else
        value = filterFunctionDefaultValue<CSS::HueRotateFunction::name>().value;

    return BasicColorMatrixFilterOperation::create(value, filterFunctionOperationType<CSS::HueRotateFunction::name>());
}

} // namespace Style
} // namespace WebCore
