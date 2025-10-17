/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

#include "CSSValueTypes.h"
#include "Color.h"
#include "ColorTypes.h"

namespace WebCore {
namespace CSS {

struct PlatformColorResolutionState;

struct HexColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    SRGBA<uint8_t> value;

    bool operator==(const HexColor&) const = default;
};

inline WebCore::Color createColor(const HexColor& unresolved, PlatformColorResolutionState&)
{
    return WebCore::Color { unresolved.value };
}

constexpr bool containsCurrentColor(const HexColor&)
{
    return false;
}

constexpr bool containsColorSchemeDependentColor(const HexColor&)
{
    return false;
}

template<> struct Serialize<HexColor> { void operator()(StringBuilder&, const HexColor&); };
template<> struct ComputedStyleDependenciesCollector<HexColor> { constexpr void operator()(ComputedStyleDependencies&, const HexColor&) { } };
template<> struct CSSValueChildrenVisitor<HexColor> { constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const HexColor&) { return IterationStatus::Continue; } };

} // namespace CSS
} // namespace WebCore
