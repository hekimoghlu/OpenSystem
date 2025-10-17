/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include "CSSColorMix.h"

#include "CSSColorMixResolver.h"
#include "CSSColorMixSerialization.h"
#include "CSSPlatformColorResolutionState.h"
#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "ColorSerialization.h"
#include "StylePrimitiveNumericTypes+Conversions.h"

namespace WebCore {
namespace CSS {

WebCore::Color createColor(const ColorMix& unresolved, PlatformColorResolutionState& state)
{
    PlatformColorResolutionStateNester nester { state };

    auto component1Color = createColor(unresolved.mixComponents1.color, state);
    if (!component1Color.isValid())
        return { };

    auto component2Color = createColor(unresolved.mixComponents2.color, state);
    if (!component2Color.isValid())
        return { };

    std::optional<Style::Percentage<Range{0, 100}>> percentage1;
    std::optional<Style::Percentage<Range{0, 100}>> percentage2;
    if (requiresConversionData(unresolved.mixComponents1.percentage) || requiresConversionData(unresolved.mixComponents2.percentage)) {
        if (!state.conversionData)
            return { };

        percentage1 = Style::toStyle(unresolved.mixComponents1.percentage, *state.conversionData);
        percentage2 = Style::toStyle(unresolved.mixComponents2.percentage, *state.conversionData);
    } else {
        percentage1 = Style::toStyleNoConversionDataRequired(unresolved.mixComponents1.percentage);
        percentage2 = Style::toStyleNoConversionDataRequired(unresolved.mixComponents2.percentage);
    }

    return mix(
        ColorMixResolver {
            unresolved.colorInterpolationMethod,
            ColorMixResolver::Component {
                WTFMove(component1Color),
                WTFMove(percentage1),
            },
            ColorMixResolver::Component {
                WTFMove(component2Color),
                WTFMove(percentage2),
            }
        }
    );
}

bool containsCurrentColor(const ColorMix& unresolved)
{
    return containsCurrentColor(unresolved.mixComponents1.color)
        || containsCurrentColor(unresolved.mixComponents2.color);
}

bool containsColorSchemeDependentColor(const ColorMix& unresolved)
{
    return containsColorSchemeDependentColor(unresolved.mixComponents1.color)
        || containsColorSchemeDependentColor(unresolved.mixComponents2.color);
}

void Serialize<ColorMix>::operator()(StringBuilder& builder, const ColorMix& value)
{
    serializationForCSSColorMix(builder, value);
}

void ComputedStyleDependenciesCollector<ColorMix>::operator()(ComputedStyleDependencies& dependencies, const ColorMix& value)
{
    collectComputedStyleDependencies(dependencies, value.mixComponents1.color);
    collectComputedStyleDependencies(dependencies, value.mixComponents1.percentage);
    collectComputedStyleDependencies(dependencies, value.mixComponents2.color);
    collectComputedStyleDependencies(dependencies, value.mixComponents2.percentage);
}

IterationStatus CSSValueChildrenVisitor<ColorMix>::operator()(const Function<IterationStatus(CSSValue&)>& func, const ColorMix& value)
{
    if (visitCSSValueChildren(func, value.mixComponents1.color) == IterationStatus::Done)
        return IterationStatus::Done;
    if (visitCSSValueChildren(func, value.mixComponents1.percentage) == IterationStatus::Done)
        return IterationStatus::Done;
    if (visitCSSValueChildren(func, value.mixComponents2.color) == IterationStatus::Done)
        return IterationStatus::Done;
    if (visitCSSValueChildren(func, value.mixComponents2.percentage) == IterationStatus::Done)
        return IterationStatus::Done;
    return IterationStatus::Continue;
}

} // namespace CSS
} // namespace WebCore
