/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include "config.h"
#include "CSSColorMixResolver.h"

#include "ColorInterpolation.h"

namespace WebCore {
namespace CSS {

namespace {

struct ColorMixPercentages {
    double p1;
    double p2;
    std::optional<double> alphaMultiplier = std::nullopt;
};

}

static std::optional<ColorMixPercentages> normalizedMixPercentages(std::optional<ColorMixResolver::Component::Percentage> mixComponents1Percentage, std::optional<ColorMixResolver::Component::Percentage> mixComponents2Percentage)
{
    // The percentages are normalized as follows:

    // 1. Let p1 be the first percentage and p2 the second one.

    // 2. If both percentages are omitted, they each default to 50% (an equal mix of the two colors).
    if (!mixComponents1Percentage && !mixComponents2Percentage)
        return {{ 50.0, 50.0 }};

    ColorMixPercentages result;

    if (!mixComponents2Percentage) {
        // 3. Otherwise, if p2 is omitted, it becomes 100% - p1
        result.p1 = mixComponents1Percentage->value;
        result.p2 = 100.0 - result.p1;
    } else if (!mixComponents1Percentage) {
        // 4. Otherwise, if p1 is omitted, it becomes 100% - p2
        result.p2 = mixComponents2Percentage->value;
        result.p1 = 100.0 - result.p2;
    } else {
        result.p1 = mixComponents1Percentage->value;
        result.p2 = mixComponents2Percentage->value;
    }

    auto sum = result.p1 + result.p2;

    // 5.If the percentages sum to zero, the function is invalid.
    if (sum == 0)
        return { };

    if (sum > 100.0) {
        // 6. Otherwise, if both are provided but do not add up to 100%, they are scaled accordingly so that they
        //    add up to 100%.
        result.p1 *= 100.0 / sum;
        result.p2 *= 100.0 / sum;
    } else if (sum < 100.0) {
        // 7. Otherwise, if both are provided and add up to less than 100%, the sum is saved as an alpha multiplier.
        //    They are then scaled accordingly so that they add up to 100%.
        result.p1 *= 100.0 / sum;
        result.p2 *= 100.0 / sum;
        result.alphaMultiplier = sum;
    }

    return result;
}

template<typename InterpolationMethod> static WebCore::Color mixColorComponentsUsingColorInterpolationMethod(InterpolationMethod interpolationMethod, ColorMixPercentages mixPercentages, const WebCore::Color& color1, const WebCore::Color& color2)
{
    using ColorType = typename InterpolationMethod::ColorType;

    // 1. Both colors are converted to the specified <color-space>. If the specified color space has a smaller gamut than
    //    the one in which the color to be adjusted is specified, gamut mapping will occur.
    auto convertedColor1 = color1.template toColorTypeLossyCarryingForwardMissing<ColorType>();
    auto convertedColor2 = color2.template toColorTypeLossyCarryingForwardMissing<ColorType>();

    // 2. Colors are then interpolated in the specified color space, as described in CSS Color 4 Â§â€¯13 Interpolation. [...]
    auto mixedColor = interpolateColorComponents<AlphaPremultiplication::Premultiplied>(interpolationMethod, convertedColor1, mixPercentages.p1 / 100.0, convertedColor2, mixPercentages.p2 / 100.0).unresolved();

    // 3. If an alpha multiplier was produced during percentage normalization, the alpha component of the interpolated result
    //    is multiplied by the alpha multiplier.
    if (mixPercentages.alphaMultiplier && !std::isnan(mixedColor.alpha))
        mixedColor.alpha *= (*mixPercentages.alphaMultiplier / 100.0);

    auto flags = OptionSet { WebCore::Color::Flags::UseColorFunctionSerialization };
    if (color1.isSemantic() || color2.isSemantic())
        flags.add(WebCore::Color::Flags::Semantic);

    return { mixedColor, flags };
}

WebCore::Color mix(const ColorMixResolver& colorMix)
{
    auto mixPercentages = normalizedMixPercentages(colorMix.mixComponents1.percentage, colorMix.mixComponents2.percentage);
    if (!mixPercentages)
        return { };

    return WTF::switchOn(colorMix.colorInterpolationMethod.colorSpace,
        [&](const auto& methodColorSpace) {
            return mixColorComponentsUsingColorInterpolationMethod(
                methodColorSpace,
                *mixPercentages,
                colorMix.mixComponents1.color,
                colorMix.mixComponents2.color
            );
        }
    );
}

} // namespace CSS
} // namespace WebCore
