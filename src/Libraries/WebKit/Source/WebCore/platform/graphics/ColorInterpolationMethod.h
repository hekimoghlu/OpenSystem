/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#include "ColorTypes.h"
#include <optional>
#include <variant>
#include <wtf/Hasher.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

enum class HueInterpolationMethod : uint8_t {
    Shorter,
    Longer,
    Increasing,
    Decreasing
};

enum class ColorInterpolationColorSpace : uint8_t {
    HSL,
    HWB,
    LCH,
    Lab,
    OKLCH,
    OKLab,
    SRGB,
    SRGBLinear,
    DisplayP3,
    A98RGB,
    ProPhotoRGB,
    Rec2020,
    XYZD50,
    XYZD65
};

template <typename T, typename = void>
struct HasHueInterpolationMethod : std::false_type { };

template <typename T>
struct HasHueInterpolationMethod<T, std::void_t<decltype(std::declval<T>().hueInterpolationMethod)>> : std::true_type { };

template <typename T>
inline constexpr bool hasHueInterpolationMethod = HasHueInterpolationMethod<T>::value;

struct ColorInterpolationMethod {
    struct HSL {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::HSL;
        using ColorType = WebCore::HSLA<float>;
        HueInterpolationMethod hueInterpolationMethod = HueInterpolationMethod::Shorter;

        friend constexpr bool operator==(const HSL&, const HSL&) = default;
    };
    struct HWB {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::HWB;
        using ColorType = WebCore::HWBA<float>;
        HueInterpolationMethod hueInterpolationMethod = HueInterpolationMethod::Shorter;

        friend constexpr bool operator==(const HWB&, const HWB&) = default;
    };
    struct LCH {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::LCH;
        using ColorType = WebCore::LCHA<float>;
        HueInterpolationMethod hueInterpolationMethod = HueInterpolationMethod::Shorter;

        friend constexpr bool operator==(const LCH&, const LCH&) = default;
    };
    struct Lab {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::Lab;
        using ColorType = WebCore::Lab<float>;
        friend constexpr bool operator==(const Lab&, const Lab&) = default;
    };
    struct OKLCH {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::OKLCH;
        using ColorType = WebCore::OKLCHA<float>;
        HueInterpolationMethod hueInterpolationMethod = HueInterpolationMethod::Shorter;
        friend constexpr bool operator==(const OKLCH&, const OKLCH&) = default;
    };
    struct OKLab {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::OKLab;
        using ColorType = WebCore::OKLab<float>;
        friend constexpr bool operator==(const OKLab&, const OKLab&) = default;
    };
    struct SRGB {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::SRGB;
        using ColorType = WebCore::ExtendedSRGBA<float>;
        friend constexpr bool operator==(const SRGB&, const SRGB&) = default;
    };
    struct SRGBLinear {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::SRGBLinear;
        using ColorType = WebCore::ExtendedLinearSRGBA<float>;
        friend constexpr bool operator==(const SRGBLinear&, const SRGBLinear&) = default;
    };
    struct DisplayP3 {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::DisplayP3;
        using ColorType = WebCore::ExtendedDisplayP3<float>;
        friend constexpr bool operator==(const DisplayP3&, const DisplayP3&) = default;
    };
    struct A98RGB {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::A98RGB;
        using ColorType = WebCore::ExtendedA98RGB<float>;
        friend constexpr bool operator==(const A98RGB&, const A98RGB&) = default;
    };
    struct ProPhotoRGB {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::ProPhotoRGB;
        using ColorType = WebCore::ExtendedProPhotoRGB<float>;
        friend constexpr bool operator==(const ProPhotoRGB&, const ProPhotoRGB&) = default;
    };
    struct Rec2020 {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::Rec2020;
        using ColorType = WebCore::ExtendedRec2020<float>;
        friend constexpr bool operator==(const Rec2020&, const Rec2020&) = default;
    };
    struct XYZD50 {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::XYZD50;
        using ColorType = WebCore::XYZA<float, WhitePoint::D50>;
        friend constexpr bool operator==(const XYZD50&, const XYZD50&) = default;
    };
    struct XYZD65 {
        static constexpr auto interpolationColorSpace = ColorInterpolationColorSpace::XYZD65;
        using ColorType = WebCore::XYZA<float, WhitePoint::D65>;
        friend constexpr bool operator==(const XYZD65&, const XYZD65&) = default;
    };

    friend constexpr bool operator==(const ColorInterpolationMethod&, const ColorInterpolationMethod&) = default;

    std::variant<
        HSL,
        HWB,
        LCH,
        Lab,
        OKLCH,
        OKLab,
        SRGB,
        SRGBLinear,
        DisplayP3,
        A98RGB,
        ProPhotoRGB,
        Rec2020,
        XYZD50,
        XYZD65
    > colorSpace;
    AlphaPremultiplication alphaPremultiplication;
};

inline void add(Hasher& hasher, const ColorInterpolationMethod& colorInterpolationMethod)
{
    add(hasher, colorInterpolationMethod.alphaPremultiplication);
    WTF::switchOn(colorInterpolationMethod.colorSpace,
        [&]<typename MethodColorSpace> (const MethodColorSpace& mothodColorSpace) {
            add(hasher, mothodColorSpace.interpolationColorSpace);
            if constexpr (MethodColorSpace::ColorType::Model::coordinateSystem == ColorSpaceCoordinateSystem::CylindricalPolar) {
                add(hasher, mothodColorSpace.hueInterpolationMethod);
            }
        }
    );
}

static constexpr ASCIILiteral serializationForCSS(ColorInterpolationColorSpace interpolationColorSpace)
{
    switch (interpolationColorSpace) {
    case ColorInterpolationColorSpace::HSL:
        return "hsl"_s;
    case ColorInterpolationColorSpace::HWB:
        return "hwb"_s;
    case ColorInterpolationColorSpace::LCH:
        return "lch"_s;
    case ColorInterpolationColorSpace::Lab:
        return "lab"_s;
    case ColorInterpolationColorSpace::OKLCH:
        return "oklch"_s;
    case ColorInterpolationColorSpace::OKLab:
        return "oklab"_s;
    case ColorInterpolationColorSpace::SRGB:
        return "srgb"_s;
    case ColorInterpolationColorSpace::SRGBLinear:
        return "srgb-linear"_s;
    case ColorInterpolationColorSpace::DisplayP3:
        return "display-p3"_s;
    case ColorInterpolationColorSpace::A98RGB:
        return "a98-rgb"_s;
    case ColorInterpolationColorSpace::ProPhotoRGB:
        return "prophoto-rgb"_s;
    case ColorInterpolationColorSpace::Rec2020:
        return "rec2020"_s;
    case ColorInterpolationColorSpace::XYZD50:
        return "xyz-d50"_s;
    case ColorInterpolationColorSpace::XYZD65:
        return "xyz-d65"_s;
    }

    ASSERT_NOT_REACHED();
    return ""_s;
}

void serializationForCSS(StringBuilder&, ColorInterpolationColorSpace);
void serializationForCSS(StringBuilder&, HueInterpolationMethod);
void serializationForCSS(StringBuilder&, const ColorInterpolationMethod&);
String serializationForCSS(const ColorInterpolationMethod&);

WTF::TextStream& operator<<(WTF::TextStream&, ColorInterpolationColorSpace);
WTF::TextStream& operator<<(WTF::TextStream&, HueInterpolationMethod);
WTF::TextStream& operator<<(WTF::TextStream&, const ColorInterpolationMethod&);

}
