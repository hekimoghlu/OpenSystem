/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include <wtf/Forward.h>

namespace WebCore {
namespace CSS {

struct PlatformColorResolutionState;

struct ResolvedColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    WebCore::Color value;

    bool operator==(const ResolvedColor&) const = default;
};

inline WebCore::Color createColor(const ResolvedColor& unresolved, PlatformColorResolutionState&)
{
    return unresolved.value;
}

constexpr bool containsColorSchemeDependentColor(const ResolvedColor&)
{
    return false;
}

constexpr bool containsCurrentColor(const ResolvedColor&)
{
    return false;
}

template<> struct Serialize<ResolvedColor> { void operator()(StringBuilder&, const ResolvedColor&); };
template<> struct ComputedStyleDependenciesCollector<ResolvedColor> { constexpr void operator()(ComputedStyleDependencies&, const ResolvedColor&) { } };
template<> struct CSSValueChildrenVisitor<ResolvedColor> { constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const ResolvedColor&) { return IterationStatus::Continue; } };

} // namespace CSS
} // namespace WebCore
