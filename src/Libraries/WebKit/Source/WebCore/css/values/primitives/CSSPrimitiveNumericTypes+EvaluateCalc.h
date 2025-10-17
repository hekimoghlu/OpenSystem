/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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

#include "CSSNoConversionDataRequiredToken.h"
#include "CSSPrimitiveNumericTypes.h"
#include "CSSUnevaluatedCalc.h"

namespace WebCore {
namespace CSS {

// MARK: - Requires Conversion Data

inline bool requiresConversionData(Numeric auto const& primitive)
{
    return WTF::switchOn(primitive, [&](const auto& value) { return requiresConversionData(value); });
}

// MARK: - Evaluation

// FIXME: Remove "evaluateCalc" family of functions once color code has moved to the "toStyle" family of functions.

template<Calc T> auto evaluateCalc(const T& calc, NoConversionDataRequiredToken token, const CSSCalcSymbolTable& symbolTable) -> typename T::Raw
{
    return { calc.evaluate(T::category, token, symbolTable) };
}

template<typename T> constexpr auto evaluateCalc(const T& component, NoConversionDataRequiredToken, const CSSCalcSymbolTable&) -> T
{
    return component;
}

template<typename... Ts> auto evaluateCalcIfNoConversionDataRequired(const std::variant<Ts...>& component, const CSSCalcSymbolTable& symbolTable) -> std::variant<Ts...>
{
    return WTF::switchOn(component, [&](const auto& alternative) -> std::variant<Ts...> {
        if (requiresConversionData(alternative))
            return alternative;
        return evaluateCalc(alternative, NoConversionDataRequiredToken { }, symbolTable);
    });
}

template<Numeric T> auto evaluateCalcIfNoConversionDataRequired(const T& component, const CSSCalcSymbolTable& symbolTable) -> T
{
    return WTF::switchOn(component, [&](const auto& alternative) -> T {
        if (requiresConversionData(alternative))
            return { alternative };
        return { evaluateCalc(alternative, NoConversionDataRequiredToken { }, symbolTable) };
    });
}

template<typename T> decltype(auto) evaluateCalcIfNoConversionDataRequired(const std::optional<T>& component, const CSSCalcSymbolTable& symbolTable)
{
    return component ? std::make_optional(evaluateCalcIfNoConversionDataRequired(*component, symbolTable)) : std::nullopt;
}

// MARK: - Is UnevaluatedCalc

inline bool isUnevaluatedCalc(Numeric auto const& value)
{
    return WTF::switchOn(value, [&](const auto& alternative) { return isUnevaluatedCalc(alternative); });
}

} // namespace CSS
} // namespace WebCore
