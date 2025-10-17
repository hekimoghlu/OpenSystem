/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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

#include "ColorComponents.h"
#include "ColorTypes.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <math.h>

namespace WebCore {

SRGBA<float> premultiplied(const SRGBA<float>&);
SRGBA<float> unpremultiplied(const SRGBA<float>&);

SRGBA<uint8_t> premultipliedFlooring(SRGBA<uint8_t>);
SRGBA<uint8_t> premultipliedCeiling(SRGBA<uint8_t>);
SRGBA<uint8_t> unpremultiplied(SRGBA<uint8_t>);

uint8_t convertPrescaledSRGBAFloatToSRGBAByte(float);

template<typename T> T convertByteAlphaTo(uint8_t);
template<typename T> T convertFloatAlphaTo(float);

template<typename ColorType, typename Functor> auto colorByModifingEachNonAlphaComponent(const ColorType&, Functor&&);

template<typename ColorType> constexpr auto colorWithOverriddenAlpha(const ColorType&, uint8_t overrideAlpha);
template<typename ColorType> auto colorWithOverriddenAlpha(const ColorType&, float overrideAlpha);

template<typename ColorType> constexpr auto invertedColorWithOverriddenAlpha(const ColorType&, uint8_t overrideAlpha);
template<typename ColorType> auto invertedColorWithOverriddenAlpha(const ColorType&, float overrideAlpha);

template<typename ColorType, typename std::enable_if_t<UsesLabModel<ColorType> || UsesLCHModel<ColorType> || UsesOKLabModel<ColorType> || UsesOKLCHModel<ColorType> || UsesHSLModel<ColorType>>* = nullptr> constexpr bool isBlack(const ColorType&);
template<typename ColorType, typename std::enable_if_t<UsesRGBModel<ColorType>>* = nullptr> constexpr bool isBlack(const ColorType&);
template<typename ColorType, typename std::enable_if_t<UsesHWBModel<ColorType>>* = nullptr> constexpr bool isBlack(const ColorType&);
template<WhitePoint W> constexpr bool isBlack(const XYZA<float, W>&);

template<typename ColorType, typename std::enable_if_t<UsesLabModel<ColorType> || UsesLCHModel<ColorType> || UsesHSLModel<ColorType>>* = nullptr> constexpr bool isWhite(const ColorType&);
template<typename ColorType, typename std::enable_if_t<UsesOKLabModel<ColorType> || UsesOKLCHModel<ColorType>>* = nullptr> constexpr bool isWhite(const ColorType&);
template<typename ColorType, typename std::enable_if_t<UsesRGBModel<ColorType> && std::is_same_v<typename ColorType::ComponentType, float>>* = nullptr> constexpr bool isWhite(const ColorType&);
template<typename ColorType, typename std::enable_if_t<UsesRGBModel<ColorType> && std::is_same_v<typename ColorType::ComponentType, uint8_t>>* = nullptr> constexpr bool isWhite(const ColorType&);
template<typename ColorType, typename std::enable_if_t<UsesHWBModel<ColorType>>* = nullptr> constexpr bool isWhite(const ColorType&);
template<WhitePoint W> constexpr bool isWhite(const XYZA<float, W>&);

constexpr uint16_t fastMultiplyBy255(uint16_t);
constexpr uint16_t fastDivideBy255(uint16_t);


inline uint8_t convertPrescaledSRGBAFloatToSRGBAByte(float value)
{
    return std::clamp(std::lround(value), 0l, 255l);
}

template<> constexpr uint8_t convertByteAlphaTo<uint8_t>(uint8_t value)
{
    return value;
}

template<> constexpr float convertByteAlphaTo<float>(uint8_t value)
{
    return value / 255.0f;
}

template<> inline uint8_t convertFloatAlphaTo<uint8_t>(float value)
{
    return std::clamp(std::lround(value * 255.0f), 0l, 255l);
}

template<> inline float convertFloatAlphaTo<float>(float value)
{
    return clampedAlpha(value);
}

template<typename ColorType, typename Functor> auto colorByModifingEachNonAlphaComponent(const ColorType& color, Functor&& functor)
{
    auto components = asColorComponents(color.resolved());
    auto copy = components;
    copy[0] = std::invoke(functor, components[0]);
    copy[1] = std::invoke(functor, components[1]);
    copy[2] = std::invoke(std::forward<Functor>(functor), components[2]);
    return makeFromComponents<ColorType>(copy);
}

template<typename ColorType> constexpr auto colorWithOverriddenAlpha(const ColorType& color, uint8_t overrideAlpha)
{
    auto copy = color.unresolved();
    copy.alpha = convertByteAlphaTo<typename ColorType::ComponentType>(overrideAlpha);
    return copy;
}

template<typename ColorType> auto colorWithOverriddenAlpha(const ColorType& color, float overrideAlpha)
{
    auto copy = color.unresolved();
    copy.alpha = convertFloatAlphaTo<typename ColorType::ComponentType>(overrideAlpha);
    return copy;
}

template<typename ColorType> constexpr auto invertedColorWithOverriddenAlpha(const ColorType& color, uint8_t overrideAlpha)
{
    static_assert(ColorType::Model::isInvertible);

    auto components = asColorComponents(color.resolved());
    auto copy = components;

    for (unsigned i = 0; i < 3; ++i)
        copy[i] = ColorType::Model::componentInfo[i].max - components[i];
    copy[3] = convertByteAlphaTo<typename ColorType::ComponentType>(overrideAlpha);

    return makeFromComponents<ColorType>(copy);
}

template<typename ColorType> auto invertedColorWithOverriddenAlpha(const ColorType& color, float overrideAlpha)
{
    static_assert(ColorType::Model::isInvertible);

    auto components = asColorComponents(color.resolved());
    auto copy = components;

    for (unsigned i = 0; i < 3; ++i)
        copy[i] = ColorType::Model::componentInfo[i].max - components[i];
    copy[3] = convertFloatAlphaTo<typename ColorType::ComponentType>(overrideAlpha);

    return makeFromComponents<ColorType>(copy);
}

template<WhitePoint W> constexpr bool isBlack(const XYZA<float, W>& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.y == 0 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesLabModel<ColorType> || UsesLCHModel<ColorType> || UsesOKLabModel<ColorType> || UsesOKLCHModel<ColorType> || UsesHSLModel<ColorType>>*>
constexpr bool isBlack(const ColorType& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.lightness == 0 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesRGBModel<ColorType>>*>
constexpr bool isBlack(const ColorType& color)
{
    auto [c1, c2, c3, alpha] = color.resolved();
    return c1 == 0 && c2 == 0 && c3 == 0 && alpha == AlphaTraits<typename ColorType::ComponentType>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesHWBModel<ColorType>>*>
constexpr bool isBlack(const ColorType& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.blackness == 100 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

template<WhitePoint W> constexpr bool isWhite(const XYZA<float, W>& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.y == 1 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesLabModel<ColorType> || UsesLCHModel<ColorType> || UsesHSLModel<ColorType>>*>
constexpr bool isWhite(const ColorType& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.lightness == 100 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesOKLabModel<ColorType> || UsesOKLCHModel<ColorType>>*>
constexpr bool isWhite(const ColorType& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.lightness == 1 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesRGBModel<ColorType> && std::is_same_v<typename ColorType::ComponentType, float>>*>
constexpr bool isWhite(const ColorType& color)
{
    auto [c1, c2, c3, alpha] = color.resolved();
    return c1 == 1 && c2 == 1 && c3 == 1 && alpha == AlphaTraits<float>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesRGBModel<ColorType> && std::is_same_v<typename ColorType::ComponentType, uint8_t>>*>
constexpr bool isWhite(const ColorType& color)
{
    auto [c1, c2, c3, alpha] = color.resolved();
    return c1 == 255 && c2 == 255 && c3 == 255 && alpha == AlphaTraits<uint8_t>::opaque;
}

template<typename ColorType, typename std::enable_if_t<UsesHWBModel<ColorType>>*>
constexpr bool isWhite(const ColorType& color)
{
    auto resolvedColor = color.resolved();
    return resolvedColor.whiteness == 100 && resolvedColor.alpha == AlphaTraits<float>::opaque;
}

constexpr uint16_t fastMultiplyBy255(uint16_t value)
{
    return (value << 8) - value;
}

constexpr uint16_t fastDivideBy255(uint16_t value)
{
    // While this is an approximate algorithm for division by 255, it gives perfectly accurate results for 16-bit values.
    // FIXME: Since this gives accurate results for 16-bit values, we should get this optimization into compilers like clang.
    uint16_t approximation = value >> 8;
    uint16_t remainder = value - (approximation * 255) + 1;
    return approximation + (remainder >> 8);
}

} // namespace WebCore
