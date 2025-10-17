/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
#include "CSSLightDarkColor.h"

#include "CSSColor.h"
#include "CSSPlatformColorResolutionState.h"
#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "Document.h"
#include "StyleBuilderState.h"

namespace WebCore {
namespace CSS {

bool LightDarkColor::operator==(const LightDarkColor&) const = default;

WebCore::Color createColor(const LightDarkColor& unresolved, PlatformColorResolutionState& state)
{
    if (!state.appearance)
        return { };

    PlatformColorResolutionStateNester nester { state };

    switch (*state.appearance) {
    case LightDarkColorAppearance::Light:
        return createColor(unresolved.lightColor, state);
    case LightDarkColorAppearance::Dark:
        return createColor(unresolved.darkColor, state);
    }

    ASSERT_NOT_REACHED();
    return { };
}

bool containsCurrentColor(const LightDarkColor& unresolved)
{
    return containsCurrentColor(unresolved.lightColor)
        || containsCurrentColor(unresolved.darkColor);
}

void Serialize<LightDarkColor>::operator()(StringBuilder& builder, const LightDarkColor& value)
{
    builder.append("light-dark("_s);
    serializationForCSS(builder, value.lightColor);
    builder.append(", "_s);
    serializationForCSS(builder, value.darkColor);
    builder.append(')');
}

void ComputedStyleDependenciesCollector<LightDarkColor>::operator()(ComputedStyleDependencies& dependencies, const LightDarkColor& value)
{
    collectComputedStyleDependencies(dependencies, value.lightColor);
    collectComputedStyleDependencies(dependencies, value.darkColor);
}

IterationStatus CSSValueChildrenVisitor<LightDarkColor>::operator()(const Function<IterationStatus(CSSValue&)>& func, const LightDarkColor& value)
{
    if (visitCSSValueChildren(func, value.lightColor) == IterationStatus::Done)
        return IterationStatus::Done;
    if (visitCSSValueChildren(func, value.darkColor) == IterationStatus::Done)
        return IterationStatus::Done;
    return IterationStatus::Continue;
}

} // namespace CSS
} // namespace WebCore
