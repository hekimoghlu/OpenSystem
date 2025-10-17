/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#include "StyleCubicBezierEasingFunction.h"

#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "TimingFunction.h"

namespace WebCore {
namespace Style {

CSS::CubicBezierEasingFunction toCSSCubicBezierEasingFunction(const CubicBezierTimingFunction& function, const RenderStyle& style)
{
    ASSERT(function.timingFunctionPreset() == CubicBezierTimingFunction::TimingFunctionPreset::Custom);

    return CSS::CubicBezierEasingFunction {
        .parameters = {
            .value = {
                CSS::CubicBezierEasingParameters::Coordinate {
                    toCSS(Number<CSS::ClosedUnitRange> { function.x1() }, style), toCSS(Number<> { function.y1() }, style)
                },
                CSS::CubicBezierEasingParameters::Coordinate {
                    toCSS(Number<CSS::ClosedUnitRange> { function.x2() }, style), toCSS(Number<> { function.y2() }, style)
                },
            }
        }
    };
}

Ref<TimingFunction> createTimingFunction(const CSS::CubicBezierEasingFunction& function, const CSSToLengthConversionData& conversionData)
{
    return CubicBezierTimingFunction::create(
        toStyle(get<0>(get<0>(function->value)), conversionData).value,
        toStyle(get<1>(get<0>(function->value)), conversionData).value,
        toStyle(get<0>(get<1>(function->value)), conversionData).value,
        toStyle(get<1>(get<1>(function->value)), conversionData).value
    );
}

Ref<TimingFunction> createTimingFunctionDeprecated(const CSS::CubicBezierEasingFunction& function)
{
    if (!CSS::collectComputedStyleDependencies(function).canResolveDependenciesWithConversionData({ }))
        return CubicBezierTimingFunction::create();

    return CubicBezierTimingFunction::create(
        toStyleNoConversionDataRequired(get<0>(get<0>(function->value))).value,
        toStyleNoConversionDataRequired(get<1>(get<0>(function->value))).value,
        toStyleNoConversionDataRequired(get<0>(get<1>(function->value))).value,
        toStyleNoConversionDataRequired(get<1>(get<1>(function->value))).value
    );
}

} // namespace Style
} // namespace WebCore
