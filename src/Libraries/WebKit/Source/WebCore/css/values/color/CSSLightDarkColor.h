/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#pragma once

#include "CSSColor.h"

namespace WebCore {

class Color;

namespace CSS {

struct PlatformColorResolutionState;

enum class LightDarkColorAppearance : bool { Light, Dark };

struct LightDarkColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    Color lightColor;
    Color darkColor;

    bool operator==(const LightDarkColor&) const;
};

WebCore::Color createColor(const LightDarkColor&, PlatformColorResolutionState&);
bool containsCurrentColor(const LightDarkColor&);

constexpr bool containsColorSchemeDependentColor(const LightDarkColor&)
{
    return true;
}

template<> struct Serialize<LightDarkColor> { void operator()(StringBuilder&, const LightDarkColor&); };
template<> struct ComputedStyleDependenciesCollector<LightDarkColor> { void operator()(ComputedStyleDependencies&, const LightDarkColor&); };
template<> struct CSSValueChildrenVisitor<LightDarkColor> { IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const LightDarkColor&); };

} // namespace CSS
} // namespace WebCore
