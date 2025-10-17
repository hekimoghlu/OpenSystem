/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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

#include "AlphaPremultiplication.h"
#include "ColorInterpolationMethod.h"
#include "ColorNormalization.h"
#include "ColorTypes.h"

namespace WebCore {

// Provides support for premulitplied and unpremultiplied operations on two color
// operands. https://www.w3.org/TR/css-values-4/#combine-colors

class Color;

template<AlphaPremultiplication, typename InterpolationMethodColorSpace>
typename InterpolationMethodColorSpace::ColorType interpolateColorComponents(InterpolationMethodColorSpace, typename InterpolationMethodColorSpace::ColorType color1, double color1Multiplier, typename InterpolationMethodColorSpace::ColorType color2, double color2Multiplier);

Color interpolateColors(ColorInterpolationMethod, Color color1, double color1Multiplier, Color color2, double color2Multiplier);

template<AlphaPremultiplication, typename InterpolationMethodColorSpace>
typename InterpolationMethodColorSpace::ColorType addColorComponents(InterpolationMethodColorSpace, typename InterpolationMethodColorSpace::ColorType color1, typename InterpolationMethodColorSpace::ColorType color2);

// MARK: - Pre-interpolation normalization/fixup.

std::pair<float, float> fixupHueComponentsPriorToInterpolation(HueInterpolationMethod, float component1, float component2);

// MARK: - Premultiplication-agnostic interpolation helpers.

inline float interpolateComponentWithoutAccountingForNaN(float componentFromColor1, double color1Multiplier, float componentFromColor2, double color2Multiplier)
{
    return (componentFromColor1 * color1Multiplier) + (componentFromColor2 * color2Multiplier);
}

inline float interpolateComponentAccountingForNaN(float componentFromColor1, double color1Multiplier, float componentFromColor2, double color2Multiplier)
{
    if (std::isnan(componentFromColor1))
        return componentFromColor2;
    if (std::isnan(componentFromColor2))
        return componentFromColor1;

    return interpolateComponentWithoutAccountingForNaN(componentFromColor1, color1Multiplier, componentFromColor2, color2Multiplier);
}

template<typename InterpolationMethodColorSpace>
float interpolateHue(InterpolationMethodColorSpace interpolationMethodColorSpace, float componentFromColor1, double color1Multiplier, float componentFromColor2, double color2Multiplier)
{
    if (std::isnan(componentFromColor1))
        return componentFromColor2;
    if (std::isnan(componentFromColor2))
        return componentFromColor1;

    auto [fixedupComponent1, fixedupComponent2] = fixupHueComponentsPriorToInterpolation(interpolationMethodColorSpace.hueInterpolationMethod, componentFromColor1, componentFromColor2);
    return interpolateComponentWithoutAccountingForNaN(fixedupComponent1, color1Multiplier, fixedupComponent2, color2Multiplier);
}

// MARK: - Premultiplied interpolation.

struct PremultipliedAlphaState {
    float alphaForPremultiplicationOfColor1;
    float alphaForPremultiplicationOfColor2;
    float alphaForUnpremultiplication;
    float resultAlpha;
};
inline PremultipliedAlphaState interpolateAlphaPremulitplied(float alphaForColor1, double color1Multiplier, float alphaForColor2, double color2Multiplier)
{
    // If both alpha channels are none/missing, no premultiplication is performed and the resulting color will have a none/missing alpha channel.
    // If only one alpha channels is none/missing, the other alpha channel is used premultiplication of both colors and is the resulting color's alpha channel.
    // If neither alpha channel is none/missing, each alpha channel is used for the premultiplication of its associated color and the interpolated result of the two alpha channels is the resulting color's alpha channel.

    if (std::isnan(alphaForColor1)) {
        if (std::isnan(alphaForColor2))
            return { 1.0f, 1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN() };
        return { alphaForColor2, alphaForColor2, alphaForColor2, alphaForColor2 };
    }
    if (std::isnan(alphaForColor2))
        return { alphaForColor1, alphaForColor1, alphaForColor1, alphaForColor1 };

    auto interpolatedAlpha = std::clamp(interpolateComponentWithoutAccountingForNaN(alphaForColor1, color1Multiplier, alphaForColor2, color2Multiplier), 0.0f, 1.0f);
    return { alphaForColor1, alphaForColor2, interpolatedAlpha, interpolatedAlpha };
}

template<size_t I, typename InterpolationMethodColorSpace>
float interpolateComponentUsingPremultipliedAlpha(InterpolationMethodColorSpace interpolationMethodColorSpace, ColorComponents<float, 4> colorComponents1, double color1Multiplier, ColorComponents<float, 4> colorComponents2, double color2Multiplier, PremultipliedAlphaState interpolatedAlpha)
{
    using ColorType = typename InterpolationMethodColorSpace::ColorType;
    constexpr auto componentInfo = ColorType::Model::componentInfo;

    if constexpr (componentInfo[I].type == ColorComponentType::Angle)
        return interpolateHue(interpolationMethodColorSpace, colorComponents1[I], color1Multiplier, colorComponents2[I], color2Multiplier);
    else {
        if (std::isnan(colorComponents1[I]))
            return colorComponents2[I];
        if (std::isnan(colorComponents2[I]))
            return colorComponents1[I];

        auto premultipliedComponent1 = colorComponents1[I] * interpolatedAlpha.alphaForPremultiplicationOfColor1;
        auto premultipliedComponent2 = colorComponents2[I] * interpolatedAlpha.alphaForPremultiplicationOfColor2;

        auto premultipliedResult = interpolateComponentWithoutAccountingForNaN(premultipliedComponent1, color1Multiplier, premultipliedComponent2, color2Multiplier);

        if (interpolatedAlpha.alphaForUnpremultiplication == 0.0f)
            return premultipliedResult;
        return premultipliedResult / interpolatedAlpha.alphaForUnpremultiplication;
    }
}

// MARK: - Unpremultiplied interpolation.

inline float interpolateAlphaUnpremulitplied(float alphaForColor1, double color1Multiplier, float alphaForColor2, double color2Multiplier)
{
    return interpolateComponentAccountingForNaN(alphaForColor1, color1Multiplier, alphaForColor2, color2Multiplier);
}

template<size_t I, typename InterpolationMethodColorSpace>
float interpolateComponentUsingUnpremultipliedAlpha(InterpolationMethodColorSpace interpolationMethodColorSpace, ColorComponents<float, 4> colorComponents1, double color1Multiplier, ColorComponents<float, 4> colorComponents2, double color2Multiplier)
{
    using ColorType = typename InterpolationMethodColorSpace::ColorType;
    constexpr auto componentInfo = ColorType::Model::componentInfo;

    if constexpr (componentInfo[I].type == ColorComponentType::Angle)
        return interpolateHue(interpolationMethodColorSpace, colorComponents1[I], color1Multiplier, colorComponents2[I], color2Multiplier);
    else
        return interpolateComponentAccountingForNaN(colorComponents1[I], color1Multiplier, colorComponents2[I], color2Multiplier);
}

// MARK: - Interpolation.

template<AlphaPremultiplication alphaPremultiplication, typename InterpolationMethodColorSpace>
typename InterpolationMethodColorSpace::ColorType interpolateColorComponents(InterpolationMethodColorSpace interpolationMethodColorSpace, typename InterpolationMethodColorSpace::ColorType color1, double color1Multiplier, typename InterpolationMethodColorSpace::ColorType color2, double color2Multiplier)
{
    auto colorComponents1 = asColorComponents(color1.unresolved());
    auto colorComponents2 = asColorComponents(color2.unresolved());

    if constexpr (alphaPremultiplication == AlphaPremultiplication::Premultiplied) {
        auto interpolatedAlpha = interpolateAlphaPremulitplied(colorComponents1[3], color1Multiplier, colorComponents2[3], color2Multiplier);
        auto interpolatedComponent1 = interpolateComponentUsingPremultipliedAlpha<0>(interpolationMethodColorSpace, colorComponents1, color1Multiplier, colorComponents2, color2Multiplier, interpolatedAlpha);
        auto interpolatedComponent2 = interpolateComponentUsingPremultipliedAlpha<1>(interpolationMethodColorSpace, colorComponents1, color1Multiplier, colorComponents2, color2Multiplier, interpolatedAlpha);
        auto interpolatedComponent3 = interpolateComponentUsingPremultipliedAlpha<2>(interpolationMethodColorSpace, colorComponents1, color1Multiplier, colorComponents2, color2Multiplier, interpolatedAlpha);

        return makeColorTypeByNormalizingComponents<typename InterpolationMethodColorSpace::ColorType>({ interpolatedComponent1, interpolatedComponent2, interpolatedComponent3, interpolatedAlpha.resultAlpha });
    } else {
        auto interpolatedAlpha = interpolateAlphaUnpremulitplied(colorComponents1[3], color1Multiplier, colorComponents2[3], color2Multiplier);
        auto interpolatedComponent1 = interpolateComponentUsingUnpremultipliedAlpha<0>(interpolationMethodColorSpace, colorComponents1, color1Multiplier, colorComponents2, color2Multiplier);
        auto interpolatedComponent2 = interpolateComponentUsingUnpremultipliedAlpha<1>(interpolationMethodColorSpace, colorComponents1, color1Multiplier, colorComponents2, color2Multiplier);
        auto interpolatedComponent3 = interpolateComponentUsingUnpremultipliedAlpha<2>(interpolationMethodColorSpace, colorComponents1, color1Multiplier, colorComponents2, color2Multiplier);

        return makeColorTypeByNormalizingComponents<typename InterpolationMethodColorSpace::ColorType>({ interpolatedComponent1, interpolatedComponent2, interpolatedComponent3, interpolatedAlpha });
    }
}

// MARK: - Addition.

template<AlphaPremultiplication alphaPremultiplication, typename InterpolationMethodColorSpace>
typename InterpolationMethodColorSpace::ColorType addColorComponents(InterpolationMethodColorSpace interpolationMethodColorSpace, typename InterpolationMethodColorSpace::ColorType color1, typename InterpolationMethodColorSpace::ColorType color2)
{
    // Color addition can utilize the existing interpolation infrastructure by
    // combining two colors at 100% strength each.
    return interpolateColorComponents<alphaPremultiplication>(interpolationMethodColorSpace, color1, 1.0, color2, 1.0);
}

}
