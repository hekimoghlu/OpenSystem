/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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

#include "CSSCalcSymbolTable.h"
#include "CSSColorConversion+ToColor.h"
#include "CSSColorConversion+ToTypedColor.h"
#include "CSSColorDescriptors.h"
#include "CSSPrimitiveNumericTypes+SymbolReplacement.h"
#include "Color.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include <optional>

namespace WebCore {
namespace CSS {

template<typename D>
struct RelativeColorResolver {
    using Descriptor = D;

    WebCore::Color origin;
    CSSColorParseTypeWithCalcAndSymbols<Descriptor> components;
};

template<typename Descriptor>
WebCore::Color resolve(const RelativeColorResolver<Descriptor>& relative, const CSSToLengthConversionData& conversionData)
{
    auto originColor = relative.origin;
    auto originColorAsColorType = originColor.template toColorTypeLossy<GetColorType<Descriptor>>();
    auto originComponents = asColorComponents(originColorAsColorType.resolved());

    const CSSCalcSymbolTable symbolTable {
        { std::get<0>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[0] * std::get<0>(Descriptor::components).symbolMultiplier },
        { std::get<1>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[1] * std::get<1>(Descriptor::components).symbolMultiplier },
        { std::get<2>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[2] * std::get<2>(Descriptor::components).symbolMultiplier },
        { std::get<3>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[3] * std::get<3>(Descriptor::components).symbolMultiplier }
    };

    // Replace any symbol value (e.g. CSSValueR) to their corresponding channel value.
    auto componentsWithUnevaluatedCalc = CSSColorParseTypeWithCalc<Descriptor> {
        replaceSymbol(std::get<0>(relative.components), symbolTable),
        replaceSymbol(std::get<1>(relative.components), symbolTable),
        replaceSymbol(std::get<2>(relative.components), symbolTable),
        replaceSymbol(std::get<3>(relative.components), symbolTable)
    };

    // Evaluated any calc values to their corresponding channel value.
    auto components = StyleColorParseType<Descriptor> {
        Style::toStyle(std::get<0>(componentsWithUnevaluatedCalc), conversionData, symbolTable),
        Style::toStyle(std::get<1>(componentsWithUnevaluatedCalc), conversionData, symbolTable),
        Style::toStyle(std::get<2>(componentsWithUnevaluatedCalc), conversionData, symbolTable),
        Style::toStyle(std::get<3>(componentsWithUnevaluatedCalc), conversionData, symbolTable)
    };

    // Normalize values into their numeric form, forming a validated typed color.
    auto typedColor = convertToTypedColor<Descriptor>(components, originColorAsColorType.unresolved().alpha);

    // Convert the validated typed color into a `Color`,
    return convertToColor<Descriptor, CSSColorFunctionForm::Relative>(typedColor);
}

// This resolve function should only be called if the components have been checked and don't require conversion data to be resolved.
template<typename Descriptor>
WebCore::Color resolveNoConversionDataRequired(const RelativeColorResolver<Descriptor>& relative)
{
    ASSERT(!componentsRequireConversionData<Descriptor>(relative.components));

    auto originColor = relative.origin;
    auto originColorAsColorType = originColor.template toColorTypeLossy<GetColorType<Descriptor>>();
    auto originComponents = asColorComponents(originColorAsColorType.resolved());

    const CSSCalcSymbolTable symbolTable {
        { std::get<0>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[0] * std::get<0>(Descriptor::components).symbolMultiplier },
        { std::get<1>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[1] * std::get<1>(Descriptor::components).symbolMultiplier },
        { std::get<2>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[2] * std::get<2>(Descriptor::components).symbolMultiplier },
        { std::get<3>(Descriptor::components).symbol, CSSUnitType::CSS_NUMBER, originComponents[3] * std::get<3>(Descriptor::components).symbolMultiplier }
    };

    // Replace any symbol value (e.g. CSSValueR) to their corresponding channel value.
    auto componentsWithUnevaluatedCalc = CSSColorParseTypeWithCalc<Descriptor> {
        replaceSymbol(std::get<0>(relative.components), symbolTable),
        replaceSymbol(std::get<1>(relative.components), symbolTable),
        replaceSymbol(std::get<2>(relative.components), symbolTable),
        replaceSymbol(std::get<3>(relative.components), symbolTable)
    };

    // Evaluated any calc values to their corresponding channel value.
    auto components = StyleColorParseType<Descriptor> {
        Style::toStyleNoConversionDataRequired(std::get<0>(componentsWithUnevaluatedCalc), symbolTable),
        Style::toStyleNoConversionDataRequired(std::get<1>(componentsWithUnevaluatedCalc), symbolTable),
        Style::toStyleNoConversionDataRequired(std::get<2>(componentsWithUnevaluatedCalc), symbolTable),
        Style::toStyleNoConversionDataRequired(std::get<3>(componentsWithUnevaluatedCalc), symbolTable)
    };

    // Normalize values into their numeric form, forming a validated typed color.
    auto typedColor = convertToTypedColor<Descriptor>(components, originColorAsColorType.unresolved().alpha);

    // Convert the validated typed color into a `Color`,
    return convertToColor<Descriptor, CSSColorFunctionForm::Relative>(typedColor);
}

} // namespace CSS
} // namespace WebCore
