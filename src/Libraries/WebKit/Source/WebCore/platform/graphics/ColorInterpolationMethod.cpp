/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#include "ColorInterpolationMethod.h"

#include <wtf/text/StringBuilder.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

void serializationForCSS(StringBuilder& builder, ColorInterpolationColorSpace interpolationColorSpace)
{
    builder.append(serializationForCSS(interpolationColorSpace));
}

void serializationForCSS(StringBuilder& builder, HueInterpolationMethod hueInterpolationMethod)
{
    switch (hueInterpolationMethod) {
    case HueInterpolationMethod::Shorter:
        break;
    case HueInterpolationMethod::Longer:
        builder.append(" longer hue"_s);
        break;
    case HueInterpolationMethod::Increasing:
        builder.append(" increasing hue"_s);
        break;
    case HueInterpolationMethod::Decreasing:
        builder.append(" decreasing hue"_s);
        break;
    }
}

void serializationForCSS(StringBuilder& builder, const ColorInterpolationMethod& method)
{
    WTF::switchOn(method.colorSpace,
        [&]<typename MethodColorSpace> (const MethodColorSpace& type) {
            serializationForCSS(builder, type.interpolationColorSpace);
            if constexpr (hasHueInterpolationMethod<MethodColorSpace>)
                serializationForCSS(builder, type.hueInterpolationMethod);
        }
    );
}

String serializationForCSS(const ColorInterpolationMethod& method)
{
    StringBuilder builder;
    serializationForCSS(builder, method);
    return builder.toString();
}

TextStream& operator<<(TextStream& ts, ColorInterpolationColorSpace interpolationColorSpace)
{
    switch (interpolationColorSpace) {
    case ColorInterpolationColorSpace::HSL:
        ts << "HSL";
        break;
    case ColorInterpolationColorSpace::HWB:
        ts << "HWB";
        break;
    case ColorInterpolationColorSpace::LCH:
        ts << "LCH";
        break;
    case ColorInterpolationColorSpace::Lab:
        ts << "Lab";
        break;
    case ColorInterpolationColorSpace::OKLCH:
        ts << "OKLCH";
        break;
    case ColorInterpolationColorSpace::OKLab:
        ts << "OKLab";
        break;
    case ColorInterpolationColorSpace::SRGB:
        ts << "sRGB";
        break;
    case ColorInterpolationColorSpace::SRGBLinear:
        ts << "sRGB linear";
        break;
    case ColorInterpolationColorSpace::DisplayP3:
        ts << "Display P3";
        break;
    case ColorInterpolationColorSpace::A98RGB:
        ts << "A98 RGB";
        break;
    case ColorInterpolationColorSpace::ProPhotoRGB:
        ts << "ProPhoto RGB";
        break;
    case ColorInterpolationColorSpace::Rec2020:
        ts << "Rec2020";
        break;
    case ColorInterpolationColorSpace::XYZD50:
        ts << "XYZ D50";
        break;
    case ColorInterpolationColorSpace::XYZD65:
        ts << "XYZ D65";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, HueInterpolationMethod hueInterpolationMethod)
{
    switch (hueInterpolationMethod) {
    case HueInterpolationMethod::Shorter:
        ts << "shorter";
        break;
    case HueInterpolationMethod::Longer:
        ts << "longer";
        break;
    case HueInterpolationMethod::Increasing:
        ts << "increasing";
        break;
    case HueInterpolationMethod::Decreasing:
        ts << "decreasing";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const ColorInterpolationMethod& method)
{
    WTF::switchOn(method.colorSpace,
        [&]<typename ColorSpace> (const ColorSpace& type) {
            ts << type.interpolationColorSpace;
            if constexpr (hasHueInterpolationMethod<ColorSpace>)
                ts << ' ' << type.hueInterpolationMethod;
            ts << ' ' << method.alphaPremultiplication;
        }
    );
    return ts;
}

} // namespace WebCore
