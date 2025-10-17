/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#include "CSSColorLayers.h"

#include "CSSColorLayersResolver.h"
#include "CSSColorLayersSerialization.h"
#include "CSSPlatformColorResolutionState.h"
#include "ColorSerialization.h"

namespace WebCore {
namespace CSS {

WebCore::Color createColor(const ColorLayers& value, PlatformColorResolutionState& state)
{
    PlatformColorResolutionStateNester nester { state };

    auto resolver = ColorLayersResolver {
        .blendMode = value.blendMode,
        // FIXME: This should be made into a lazy transformed range to avoid the unnecessary temporary allocation.
        .colors = value.colors.map([&](const auto& color) {
            return createColor(color, state);
        })
    };

    return blendSourceOver(WTFMove(resolver));
}

bool containsCurrentColor(const ColorLayers& value)
{
    return std::ranges::any_of(value.colors, [](const auto& color) {
        return containsCurrentColor(color);
    });
}

bool containsColorSchemeDependentColor(const ColorLayers& value)
{
    return std::ranges::any_of(value.colors, [](const auto& color) {
        return containsColorSchemeDependentColor(color);
    });
}

void Serialize<ColorLayers>::operator()(StringBuilder& builder, const ColorLayers& value)
{
    serializationForCSSColorLayers(builder, value);
}

void ComputedStyleDependenciesCollector<ColorLayers>::operator()(ComputedStyleDependencies& dependencies, const ColorLayers& value)
{
    collectComputedStyleDependenciesOnRangeLike(dependencies, value.colors);
}

IterationStatus CSSValueChildrenVisitor<ColorLayers>::operator()(const Function<IterationStatus(CSSValue&)>& func, const ColorLayers& value)
{
    return visitCSSValueChildrenOnRangeLike(func, value.colors);
}

} // namespace CSS
} // namespace WebCore
