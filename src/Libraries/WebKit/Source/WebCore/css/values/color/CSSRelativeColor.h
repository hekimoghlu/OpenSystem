/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include "CSSColorDescriptors.h"
#include "CSSPlatformColorResolutionState.h"
#include "CSSPrimitiveNumericTypes+EvaluateCalc.h"
#include "CSSRelativeColorResolver.h"
#include "CSSRelativeColorSerialization.h"
#include <variant>
#include <wtf/Forward.h>

namespace WebCore {
namespace CSS {

template<typename Descriptor, unsigned Index>
using RelativeColorComponent = GetCSSColorParseTypeWithCalcAndSymbolsComponentResult<Descriptor, Index>;

template<typename D>
struct RelativeColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    using Descriptor = D;

    Color origin;
    CSSColorParseTypeWithCalcAndSymbols<Descriptor> components;

    bool operator==(const RelativeColor<Descriptor>&) const = default;
};

template<typename Descriptor>
auto simplifyUnevaluatedCalc(const CSSColorParseTypeWithCalcAndSymbols<Descriptor>& components, const CSSToLengthConversionData& conversionData, const CSSCalcSymbolTable& symbolTable) -> CSSColorParseTypeWithCalcAndSymbols<Descriptor>
{
    return CSSColorParseTypeWithCalcAndSymbols<Descriptor> {
        simplifyUnevaluatedCalc(std::get<0>(components), conversionData, symbolTable),
        simplifyUnevaluatedCalc(std::get<1>(components), conversionData, symbolTable),
        simplifyUnevaluatedCalc(std::get<2>(components), conversionData, symbolTable),
        simplifyUnevaluatedCalc(std::get<3>(components), conversionData, symbolTable)
    };
}

template<typename Descriptor>
WebCore::Color createColor(const RelativeColor<Descriptor>& unresolved, PlatformColorResolutionState& state)
{
    PlatformColorResolutionStateNester nester { state };

    auto origin = createColor(unresolved.origin, state);
    if (!origin.isValid())
        return { };

    auto resolver = RelativeColorResolver<Descriptor> {
        .origin = WTFMove(origin),
        .components = unresolved.components
    };

    if (state.conversionData)
        return resolve(WTFMove(resolver), *state.conversionData);

    if (!componentsRequireConversionData<Descriptor>(resolver.components))
        return resolveNoConversionDataRequired(WTFMove(resolver));

    return { };
}

template<typename Descriptor>
bool containsColorSchemeDependentColor(const RelativeColor<Descriptor>& unresolved)
{
    return containsColorSchemeDependentColor(unresolved.origin);
}

template<typename Descriptor>
bool containsCurrentColor(const RelativeColor<Descriptor>& unresolved)
{
    return containsColorSchemeDependentColor(unresolved.origin);
}

template<typename D> struct Serialize<RelativeColor<D>> {
    void operator()(StringBuilder& builder, const RelativeColor<D>& value)
    {
        serializationForCSSRelativeColor(builder, value);
    }
};

template<typename D> struct ComputedStyleDependenciesCollector<RelativeColor<D>> {
    void operator()(ComputedStyleDependencies& dependencies, const RelativeColor<D>& value)
    {
        collectComputedStyleDependencies(dependencies, value.origin);
        collectComputedStyleDependencies(dependencies, std::get<0>(value.components));
        collectComputedStyleDependencies(dependencies, std::get<1>(value.components));
        collectComputedStyleDependencies(dependencies, std::get<2>(value.components));
        collectComputedStyleDependencies(dependencies, std::get<3>(value.components));
    }
};

template<typename D> struct CSSValueChildrenVisitor<RelativeColor<D>> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>& func, const RelativeColor<D>& value)
    {
        if (visitCSSValueChildren(func, value.origin) == IterationStatus::Done)
            return IterationStatus::Done;
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
