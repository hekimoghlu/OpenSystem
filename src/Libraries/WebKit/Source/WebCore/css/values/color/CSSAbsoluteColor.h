/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#include "CSSAbsoluteColorResolver.h"
#include "CSSAbsoluteColorSerialization.h"
#include "CSSColorDescriptors.h"
#include "CSSPlatformColorResolutionState.h"
#include "CSSValueTypes.h"

namespace WebCore {

class Color;

namespace CSS {

template<typename D, unsigned Index> using AbsoluteColorComponent = GetCSSColorParseTypeWithCalcComponentResult<D, Index>;

template<typename D>
struct AbsoluteColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    using Descriptor = D;

    CSSColorParseTypeWithCalc<Descriptor> components;

    bool operator==(const AbsoluteColor<Descriptor>&) const = default;
};

template<typename D> WebCore::Color createColor(const AbsoluteColor<D>& unresolved, PlatformColorResolutionState& state)
{
    PlatformColorResolutionStateNester nester { state };

    auto resolver = AbsoluteColorResolver<D> {
        .components = unresolved.components,
        .nestingLevel = state.nestingLevel
    };

    if (state.conversionData)
        return resolve(WTFMove(resolver), *state.conversionData);

    if (!componentsRequireConversionData<D>(resolver.components))
        return resolveNoConversionDataRequired(WTFMove(resolver));

    return { };
}

template<typename D> constexpr bool containsColorSchemeDependentColor(const AbsoluteColor<D>&)
{
    return false;
}

template<typename D> constexpr bool containsCurrentColor(const AbsoluteColor<D>&)
{
    return false;
}

template<typename D> struct Serialize<AbsoluteColor<D>> {
    void operator()(StringBuilder& builder, const AbsoluteColor<D>& value)
    {
        serializationForCSSAbsoluteColor(builder, value);
    }
};

template<typename D> struct ComputedStyleDependenciesCollector<AbsoluteColor<D>> {
    void operator()(ComputedStyleDependencies& dependencies, const AbsoluteColor<D>& value)
    {
        collectComputedStyleDependencies(dependencies, std::get<0>(value.components));
        collectComputedStyleDependencies(dependencies, std::get<1>(value.components));
        collectComputedStyleDependencies(dependencies, std::get<2>(value.components));
        collectComputedStyleDependencies(dependencies, std::get<3>(value.components));
    }
};

template<typename D> struct CSSValueChildrenVisitor<AbsoluteColor<D>> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>& func, const AbsoluteColor<D>& value)
    {
        if (visitCSSValueChildren(func, std::get<0>(value.components)) == IterationStatus::Done)
            return IterationStatus::Done;
        if (visitCSSValueChildren(func, std::get<1>(value.components)) == IterationStatus::Done)
            return IterationStatus::Done;
        if (visitCSSValueChildren(func, std::get<2>(value.components)) == IterationStatus::Done)
            return IterationStatus::Done;
        if (visitCSSValueChildren(func, std::get<3>(value.components)) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }
};

} // namespace CSS
} // namespace WebCore
