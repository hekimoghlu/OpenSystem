/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

#include "CSSColorConversion+Normalize.h"
#include "CSSColorDescriptors.h"

namespace WebCore {

// This file implements support for converting style resolved parsed values (e.g. tuple
// of `std::variant<Style::Number<>, Style::Percentage<>, ...>`) into typed colors (e.g. `SRGBA<float>`).

template<typename Descriptor, unsigned Index> float convertToTypeColorComponent(Style::Number<> number)
{
    constexpr auto info = std::get<Index>(Descriptor::components);
    constexpr auto multiplier = info.numberMultiplier;
    constexpr auto min = info.min * info.numberMultiplier;
    constexpr auto max = info.max * info.numberMultiplier;

    if constexpr (info.type == ColorComponentType::Angle)
        return normalizeHue(number.value);
    else if constexpr (info.min == -std::numeric_limits<double>::infinity() && info.max == std::numeric_limits<double>::infinity())
        return number.value * multiplier;
    else if constexpr (info.min == -std::numeric_limits<double>::infinity())
        return std::min(number.value * multiplier, max);
    else if constexpr (info.max == std::numeric_limits<double>::infinity())
        return std::max(number.value * multiplier, min);
    else
        return std::clamp(number.value * multiplier, min, max);
}

template<typename Descriptor, unsigned Index> float convertToTypeColorComponent(Style::Percentage<> percent)
{
    constexpr auto info = std::get<Index>(Descriptor::components);
    constexpr auto multiplier = info.percentMultiplier * info.numberMultiplier;
    constexpr auto min = info.min * info.numberMultiplier;
    constexpr auto max = info.max * info.numberMultiplier;

    if constexpr (info.min == -std::numeric_limits<double>::infinity() && info.max == std::numeric_limits<double>::infinity())
        return percent.value * multiplier;
    else if constexpr (info.min == -std::numeric_limits<double>::infinity())
        return std::min(percent.value * multiplier, max);
    else if constexpr (info.max == std::numeric_limits<double>::infinity())
        return std::max(percent.value * multiplier, min);
    else
        return std::clamp(percent.value * multiplier, min, max);
}

template<typename Descriptor, unsigned Index> float convertToTypeColorComponent(Style::Angle<> angle)
{
    constexpr auto info = std::get<Index>(Descriptor::components);
    static_assert(info.type == ColorComponentType::Angle);

    return normalizeHue(angle.value);
}

template<typename Descriptor, unsigned Index> float convertToTypeColorComponent(CSS::Keyword::None)
{
    return std::numeric_limits<double>::quiet_NaN();
}

template<typename Descriptor, unsigned Index, typename... Ts> float convertToTypeColorComponent(const std::variant<Ts...>& variant)
{
    return WTF::switchOn(variant, [](auto value) { return convertToTypeColorComponent<Descriptor, Index>(value); });
}

template<typename Descriptor, unsigned Index, typename T> float convertToTypeColorComponent(const std::optional<T>& optional, float defaultValue)
{
    return optional ? convertToTypeColorComponent<Descriptor, Index>(*optional) : defaultValue;
}

template<typename Descriptor> GetColorType<Descriptor> convertToTypedColor(StyleColorParseType<Descriptor> parsed, double defaultAlpha)
{
    return {
        convertToTypeColorComponent<Descriptor, 0>(std::get<0>(parsed)),
        convertToTypeColorComponent<Descriptor, 1>(std::get<1>(parsed)),
        convertToTypeColorComponent<Descriptor, 2>(std::get<2>(parsed)),
        convertToTypeColorComponent<Descriptor, 3>(std::get<3>(parsed), defaultAlpha)
    };
}

} // namespace WebCore
