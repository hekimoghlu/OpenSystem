/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#include "CSSPropertyParserConsumer+ColorInterpolationMethod.h"

#include "CSSParserTokenRange.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "ColorInterpolationMethod.h"
#include <wtf/SortedArrayMap.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

// MARK: - color-mix()

static std::optional<HueInterpolationMethod> consumeHueInterpolationMethod(CSSParserTokenRange& range)
{
    static constexpr std::pair<CSSValueID, HueInterpolationMethod> hueInterpolationMethodMappings[] {
        { CSSValueShorter, HueInterpolationMethod::Shorter },
        { CSSValueLonger, HueInterpolationMethod::Longer },
        { CSSValueIncreasing, HueInterpolationMethod::Increasing },
        { CSSValueDecreasing, HueInterpolationMethod::Decreasing },
    };
    static constexpr SortedArrayMap hueInterpolationMethodMap { hueInterpolationMethodMappings };

    return consumeIdentUsingMapping(range, hueInterpolationMethodMap);
}

std::optional<ColorInterpolationMethod> consumeColorInterpolationMethod(CSSParserTokenRange& args, const CSSParserContext&)
{
    // <rectangular-color-space> = srgb | srgb-linear | display-p3 | a98-rgb | prophoto-rgb | rec2020 | lab | oklab | xyz | xyz-d50 | xyz-d65
    // <polar-color-space> = hsl | hwb | lch | oklch
    // <hue-interpolation-method> = [ shorter | longer | increasing | decreasing ] hue
    // <color-interpolation-method> = in [ <rectangular-color-space> | <polar-color-space> <hue-interpolation-method>? ]

    ASSERT(args.peek().id() == CSSValueIn);
    consumeIdentRaw(args);

    auto consumePolarColorSpace = [](CSSParserTokenRange& args, auto colorInterpolationMethod) -> std::optional<ColorInterpolationMethod> {
        // Consume the color space identifier.
        args.consumeIncludingWhitespace();

        // <hue-interpolation-method> is optional, so if it is not provided, we just use the default value
        // specified in the passed in 'colorInterpolationMethod' parameter.
        auto hueInterpolationMethod = consumeHueInterpolationMethod(args);
        if (!hueInterpolationMethod)
            return {{ colorInterpolationMethod, AlphaPremultiplication::Premultiplied }};

        // If the hue-interpolation-method was provided it must be followed immediately by the 'hue' identifier.
        if (!consumeIdentRaw<CSSValueHue>(args))
            return { };

        colorInterpolationMethod.hueInterpolationMethod = *hueInterpolationMethod;

        return {{ colorInterpolationMethod, AlphaPremultiplication::Premultiplied }};
    };

    auto consumeRectangularColorSpace = [](CSSParserTokenRange& args, auto colorInterpolationMethod) -> std::optional<ColorInterpolationMethod> {
        // Consume the color space identifier.
        args.consumeIncludingWhitespace();

        return {{ colorInterpolationMethod, AlphaPremultiplication::Premultiplied }};
    };

    switch (args.peek().id()) {
    case CSSValueHsl:
        return consumePolarColorSpace(args, ColorInterpolationMethod::HSL { });
    case CSSValueHwb:
        return consumePolarColorSpace(args, ColorInterpolationMethod::HWB { });
    case CSSValueLch:
        return consumePolarColorSpace(args, ColorInterpolationMethod::LCH { });
    case CSSValueLab:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::Lab { });
    case CSSValueOklch:
        return consumePolarColorSpace(args, ColorInterpolationMethod::OKLCH { });
    case CSSValueOklab:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::OKLab { });
    case CSSValueSRGB:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::SRGB { });
    case CSSValueSrgbLinear:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::SRGBLinear { });
    case CSSValueDisplayP3:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::DisplayP3 { });
    case CSSValueA98Rgb:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::A98RGB { });
    case CSSValueProphotoRgb:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::ProPhotoRGB { });
    case CSSValueRec2020:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::Rec2020 { });
    case CSSValueXyzD50:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::XYZD50 { });
    case CSSValueXyz:
    case CSSValueXyzD65:
        return consumeRectangularColorSpace(args, ColorInterpolationMethod::XYZD65 { });
    default:
        return { };
    }
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
