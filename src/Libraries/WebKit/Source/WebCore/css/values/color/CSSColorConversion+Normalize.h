/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

#include "CSSColorDescriptors.h"
#include "CSSPrimitiveNumericTypes.h"
#include "CSSPrimitiveValue.h"
#include "ColorNormalization.h"
#include <limits>
#include <optional>

namespace WebCore {

// MARK: - normalizeAndClampNumericComponents

template<typename Descriptor, unsigned Index>
CSS::Number<> normalizeAndClampNumericComponents(CSS::NumberRaw<> number)
{
    constexpr auto info = std::get<Index>(Descriptor::components);

    if constexpr (info.type == ColorComponentType::Angle)
        return { normalizeHue(number.value) };
    else if constexpr (info.min == -std::numeric_limits<double>::infinity() && info.max == std::numeric_limits<double>::infinity())
        return { number.value };
    else if constexpr (info.min == -std::numeric_limits<double>::infinity())
        return { std::min(number.value, info.max) };
    else if constexpr (info.max == std::numeric_limits<double>::infinity())
        return { std::max(number.value, info.min) };
    else
        return { std::clamp(number.value, info.min, info.max) };
}

template<typename Descriptor, unsigned Index>
CSS::Number<> normalizeAndClampNumericComponents(CSS::PercentageRaw<> percent)
{
    constexpr auto info = std::get<Index>(Descriptor::components);

    if constexpr (info.min == -std::numeric_limits<double>::infinity() && info.max == std::numeric_limits<double>::infinity())
        return { percent.value * info.percentMultiplier };
    else if constexpr (info.min == -std::numeric_limits<double>::infinity())
        return { std::min(percent.value * info.percentMultiplier, info.max) };
    else if constexpr (info.max == std::numeric_limits<double>::infinity())
        return { std::max(percent.value * info.percentMultiplier, info.min) };
    else
        return { std::clamp(percent.value * info.percentMultiplier, info.min, info.max) };
}

template<typename Descriptor, unsigned Index>
CSS::Number<> normalizeAndClampNumericComponents(CSS::AngleRaw<> angle)
{
    constexpr auto info = std::get<Index>(Descriptor::components);
    static_assert(info.type == ColorComponentType::Angle);

    return { normalizeHue(CSS::convertToValueInUnitsOf<CSS::AngleUnit::Deg>(angle)) };
}

template<typename Descriptor, unsigned Index>
auto normalizeAndClampNumericComponentsIntoCanonicalRepresentation(const CSS::Keyword::None& none) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>
{
    return none;
}

template<typename Descriptor, unsigned Index, CSS::Numeric T>
auto normalizeAndClampNumericComponentsIntoCanonicalRepresentation(const T& value) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>
{
    return WTF::switchOn(value,
        [](const typename T::Raw& value) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index> {
            return normalizeAndClampNumericComponents<Descriptor, Index>(value);
        },
        [](const typename T::Calc& value) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index> {
            return T { value };
        }
    );
}

template<typename Descriptor, unsigned Index, typename... Ts>
auto normalizeAndClampNumericComponentsIntoCanonicalRepresentation(const std::variant<Ts...>& variant) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>
{
    return WTF::switchOn(variant, [](auto value) { return normalizeAndClampNumericComponentsIntoCanonicalRepresentation<Descriptor, Index>(value); });
}

template<typename Descriptor, unsigned Index>
auto normalizeAndClampNumericComponentsIntoCanonicalRepresentation(const std::optional<GetCSSColorParseTypeWithCalcComponentResult<Descriptor, Index>>& optional) -> std::optional<GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>>
{
    return optional ? std::make_optional(normalizeAndClampNumericComponentsIntoCanonicalRepresentation<Descriptor, Index>(*optional)) : std::nullopt;
}

// MARK: - normalizeNumericComponents

template<typename Descriptor, unsigned Index>
CSS::Number<> normalizeNumericComponents(CSS::NumberRaw<> number)
{
    constexpr auto info = std::get<Index>(Descriptor::components);

    if constexpr (info.type == ColorComponentType::Angle)
        return { normalizeHue(number.value) };
    else
        return { number.value };
}

template<typename Descriptor, unsigned Index>
CSS::Number<> normalizeNumericComponents(CSS::PercentageRaw<> percent)
{
    constexpr auto info = std::get<Index>(Descriptor::components);

    return { percent.value * info.percentMultiplier };
}

template<typename Descriptor, unsigned Index>
CSS::Number<> normalizeNumericComponents(CSS::AngleRaw<> angle)
{
    constexpr auto info = std::get<Index>(Descriptor::components);
    static_assert(info.type == ColorComponentType::Angle);

    return { normalizeHue(CSS::convertToValueInUnitsOf<CSS::AngleUnit::Deg>(angle)) };
}

template<typename Descriptor, unsigned Index>
auto normalizeNumericComponentsIntoCanonicalRepresentation(const CSS::Keyword::None& none) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>
{
    return none;
}

template<typename Descriptor, unsigned Index, CSS::Numeric T>
auto normalizeNumericComponentsIntoCanonicalRepresentation(const T& value) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>
{
    return WTF::switchOn(value,
        [](const typename T::Raw& value) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index> {
            return normalizeNumericComponents<Descriptor, Index>(value);
        },
        [](const typename T::Calc& value) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index> {
            return T { value };
        }
    );
}

template<typename Descriptor, unsigned Index, typename... Ts>
auto normalizeNumericComponentsIntoCanonicalRepresentation(const std::variant<Ts...>& variant) -> GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>
{
    return WTF::switchOn(variant, [](auto value) { return normalizeNumericComponentsIntoCanonicalRepresentation<Descriptor, Index>(value); });
}

template<typename Descriptor, unsigned Index>
auto normalizeNumericComponentsIntoCanonicalRepresentation(const std::optional<GetCSSColorParseTypeWithCalcComponentResult<Descriptor, Index>>& optional) -> std::optional<GetCSSColorParseTypeWithCalcComponentResult<typename Descriptor::Canonical, Index>>
{
    return optional ? std::make_optional(normalizeNumericComponentsIntoCanonicalRepresentation<Descriptor, Index>(*optional)) : std::nullopt;
}

} // namespace WebCore
