/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#include "Color.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include <optional>

namespace WebCore {
namespace CSS {

template<typename D> struct AbsoluteColorResolver {
    using Descriptor = D;

    CSSColorParseTypeWithCalc<Descriptor> components;
    unsigned nestingLevel;
};

template<typename D> WebCore::Color resolve(const AbsoluteColorResolver<D>& absolute, const CSSToLengthConversionData& conversionData)
{
    // Evaluated any calc values to their corresponding channel value.
    auto components = StyleColorParseType<D> {
        Style::toStyle(std::get<0>(absolute.components), conversionData),
        Style::toStyle(std::get<1>(absolute.components), conversionData),
        Style::toStyle(std::get<2>(absolute.components), conversionData),
        Style::toStyle(std::get<3>(absolute.components), conversionData)
    };

    // Normalize values into their numeric form, forming a validated typed color.
    auto typedColor = convertToTypedColor<D>(components, 1.0);

    // Convert the validated typed color into a `WebCore::Color`,
    return convertToColor<D, CSSColorFunctionForm::Absolute>(typedColor, absolute.nestingLevel);
}

// This resolve function should only be called if the components have been checked and don't require conversion data to be resolved.
template<typename D> WebCore::Color resolveNoConversionDataRequired(const AbsoluteColorResolver<D>& absolute)
{
    ASSERT(!componentsRequireConversionData<D>(absolute.components));

    // Evaluated any calc values to their corresponding channel value.
    auto components = StyleColorParseType<D> {
        Style::toStyleNoConversionDataRequired(std::get<0>(absolute.components)),
        Style::toStyleNoConversionDataRequired(std::get<1>(absolute.components)),
        Style::toStyleNoConversionDataRequired(std::get<2>(absolute.components)),
        Style::toStyleNoConversionDataRequired(std::get<3>(absolute.components))
    };

    // Normalize values into their numeric form, forming a validated typed color.
    auto typedColor = convertToTypedColor<D>(components, 1.0);

    // Convert the validated typed color into a `WebCore::Color`,
    return convertToColor<D, CSSColorFunctionForm::Absolute>(typedColor, absolute.nestingLevel);
}

} // namespace CSS
} // namespace WebCore
