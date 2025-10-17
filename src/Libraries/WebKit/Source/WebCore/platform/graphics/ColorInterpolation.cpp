/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include "ColorInterpolation.h"

#include "Color.h"

namespace WebCore {

std::pair<float, float> fixupHueComponentsPriorToInterpolation(HueInterpolationMethod method, float component1, float component2)
{
    auto normalizeAnglesUsingShorterAlgorithm = [] (auto theta1, auto theta2) -> std::pair<float, float> {
        // https://drafts.csswg.org/css-color-4/#hue-shorter
        auto difference = theta2 - theta1;
        if (difference > 180.0)
            return { theta1 + 360.0, theta2 };
        if (difference < -180.0)
            return { theta1, theta2 + 360.0 };
        return { theta1, theta2 };
    };

    auto normalizeAnglesUsingLongerAlgorithm = [] (auto theta1, auto theta2) -> std::pair<float, float> {
        // https://drafts.csswg.org/css-color-4/#hue-longer
        auto difference = theta2 - theta1;
        if (difference > 0.0 && difference < 180.0)
            return { theta1 + 360.0, theta2 };
        if (difference > -180.0 && difference <= 0)
            return { theta1, theta2 + 360.0 };
        return { theta1, theta2 };
    };

    auto normalizeAnglesUsingIncreasingAlgorithm = [] (auto theta1, auto theta2) -> std::pair<float, float> {
        // https://drafts.csswg.org/css-color-4/#hue-increasing
        if (theta2 < theta1)
            return { theta1, theta2 + 360.0 };
        return { theta1, theta2 };
    };

    auto normalizeAnglesUsingDecreasingAlgorithm = [] (auto theta1, auto theta2) -> std::pair<float, float> {
        // https://drafts.csswg.org/css-color-4/#hue-decreasing
        if (theta1 < theta2)
            return { theta1 + 360.0, theta2 };
        return { theta1, theta2 };
    };

    // https://www.w3.org/TR/css-color-4/#hue-interpolation
    //    "Both angles need to be constrained to [0, 360] prior to interpolation.
    //     One way to do this is Î¸ = ((Î¸ % 360) + 360) % 360."

    switch (method) {
    case HueInterpolationMethod::Shorter:
        return normalizeAnglesUsingShorterAlgorithm(normalizeHue(component1), normalizeHue(component2));
    case HueInterpolationMethod::Longer:
        return normalizeAnglesUsingLongerAlgorithm(normalizeHue(component1), normalizeHue(component2));
    case HueInterpolationMethod::Increasing:
        return normalizeAnglesUsingIncreasingAlgorithm(normalizeHue(component1), normalizeHue(component2));
    case HueInterpolationMethod::Decreasing:
        return normalizeAnglesUsingDecreasingAlgorithm(normalizeHue(component1), normalizeHue(component2));
    }
    RELEASE_ASSERT_NOT_REACHED();
}

Color interpolateColors(ColorInterpolationMethod colorInterpolationMethod, Color color1, double color1Multiplier, Color color2, double color2Multiplier)
{
    return WTF::switchOn(colorInterpolationMethod.colorSpace,
        [&]<typename MethodColorSpace> (const MethodColorSpace& colorSpace) -> Color {
            using ColorType = typename MethodColorSpace::ColorType;
            switch (colorInterpolationMethod.alphaPremultiplication) {
            case AlphaPremultiplication::Premultiplied:
                return interpolateColorComponents<AlphaPremultiplication::Premultiplied>(
                    colorSpace,
                    color1.toColorTypeLossyCarryingForwardMissing<ColorType>(), color1Multiplier,
                    color2.toColorTypeLossyCarryingForwardMissing<ColorType>(), color2Multiplier
                );

            case AlphaPremultiplication::Unpremultiplied:
                return interpolateColorComponents<AlphaPremultiplication::Unpremultiplied>(
                    colorSpace,
                    color1.toColorTypeLossyCarryingForwardMissing<ColorType>(), color1Multiplier,
                    color2.toColorTypeLossyCarryingForwardMissing<ColorType>(), color2Multiplier
                );
            }
            RELEASE_ASSERT_NOT_REACHED();
        }
    );
}

}
