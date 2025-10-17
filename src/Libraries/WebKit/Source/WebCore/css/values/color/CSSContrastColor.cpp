/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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
#include "CSSContrastColor.h"

#include "CSSColor.h"
#include "CSSContrastColorResolver.h"
#include "CSSContrastColorSerialization.h"
#include "CSSPlatformColorResolutionState.h"
#include "ColorSerialization.h"

namespace WebCore {
namespace CSS {

bool ContrastColor::operator==(const ContrastColor&) const = default;

WebCore::Color createColor(const ContrastColor& unresolved, PlatformColorResolutionState& state)
{
    PlatformColorResolutionStateNester nester { state };

    return resolve(
        ContrastColorResolver {
            createColor(unresolved.color, state),
            unresolved.max
        }
    );
}

bool containsCurrentColor(const ContrastColor& unresolved)
{
    return containsCurrentColor(unresolved.color);
}

bool containsColorSchemeDependentColor(const ContrastColor& unresolved)
{
    return containsColorSchemeDependentColor(unresolved.color);
}

void Serialize<ContrastColor>::operator()(StringBuilder& builder, const ContrastColor& value)
{
    serializationForCSSContrastColor(builder, value);
}

void ComputedStyleDependenciesCollector<ContrastColor>::operator()(ComputedStyleDependencies& dependencies, const ContrastColor& value)
{
    collectComputedStyleDependencies(dependencies, value.color);
}

IterationStatus CSSValueChildrenVisitor<ContrastColor>::operator()(const Function<IterationStatus(CSSValue&)>& func, const ContrastColor& value)
{
    return visitCSSValueChildren(func, value.color);
}

} // namespace CSS
} // namespace WebCore
