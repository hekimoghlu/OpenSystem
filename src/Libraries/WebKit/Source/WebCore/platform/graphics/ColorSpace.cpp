/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#include "ColorSpace.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, ColorSpace colorSpace)
{
    switch (colorSpace) {
    case ColorSpace::A98RGB:
        ts << "A98-RGB";
        break;
    case ColorSpace::DisplayP3:
        ts << "DisplayP3";
        break;
    case ColorSpace::ExtendedA98RGB:
        ts << "Extended A98-RGB";
        break;
    case ColorSpace::ExtendedDisplayP3:
        ts << "Extended DisplayP3";
        break;
    case ColorSpace::ExtendedLinearSRGB:
        ts << "Extended Linear sRGB";
        break;
    case ColorSpace::ExtendedProPhotoRGB:
        ts << "Extended ProPhotoRGB";
        break;
    case ColorSpace::ExtendedRec2020:
        ts << "Extended Rec2020";
        break;
    case ColorSpace::ExtendedSRGB:
        ts << "Extended sRGB";
        break;
    case ColorSpace::HSL:
        ts << "HSL";
        break;
    case ColorSpace::HWB:
        ts << "HWB";
        break;
    case ColorSpace::LCH:
        ts << "LCH";
        break;
    case ColorSpace::Lab:
        ts << "Lab";
        break;
    case ColorSpace::LinearSRGB:
        ts << "Linear sRGB";
        break;
    case ColorSpace::OKLCH:
        ts << "OKLCH";
        break;
    case ColorSpace::OKLab:
        ts << "OKLab";
        break;
    case ColorSpace::ProPhotoRGB:
        ts << "ProPhotoRGB";
        break;
    case ColorSpace::Rec2020:
        ts << "Rec2020";
        break;
    case ColorSpace::SRGB:
        ts << "sRGB";
        break;
    case ColorSpace::XYZ_D50:
        ts << "XYZ-D50";
        break;
    case ColorSpace::XYZ_D65:
        ts << "XYZ-D50";
        break;
    }
    return ts;
}

}
