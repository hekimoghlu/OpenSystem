/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "StyleSpringEasingFunction.h"

#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "TimingFunction.h"

namespace WebCore {
namespace Style {

CSS::SpringEasingFunction toCSSSpringEasingFunction(const SpringTimingFunction& function, const RenderStyle& style)
{
    return CSS::SpringEasingFunction {
        .parameters = {
            .mass = toCSS(Number<CSS::SpringEasingParameters::Positive> { function.mass() }, style),
            .stiffness = toCSS(Number<CSS::SpringEasingParameters::Positive> { function.stiffness() }, style),
            .damping = toCSS(Number<CSS::Nonnegative> { function.damping() }, style),
            .initialVelocity = toCSS(Number<> { function.initialVelocity() }, style),
        }
    };
}

Ref<TimingFunction> createTimingFunction(const CSS::SpringEasingFunction& function, const CSSToLengthConversionData& conversionData)
{
    return SpringTimingFunction::create(
        toStyle(function->mass, conversionData).value,
        toStyle(function->stiffness, conversionData).value,
        toStyle(function->damping, conversionData).value,
        toStyle(function->initialVelocity, conversionData).value
    );
}

Ref<TimingFunction> createTimingFunctionDeprecated(const CSS::SpringEasingFunction& function)
{
    if (!CSS::collectComputedStyleDependencies(function).canResolveDependenciesWithConversionData({ }))
        return SpringTimingFunction::create(1, 1, 0, 0);

    return SpringTimingFunction::create(
        toStyleNoConversionDataRequired(function->mass).value,
        toStyleNoConversionDataRequired(function->stiffness).value,
        toStyleNoConversionDataRequired(function->damping).value,
        toStyleNoConversionDataRequired(function->initialVelocity).value
    );
}

} // namespace Style
} // namespace WebCore
