/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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

#include "ColorConversion.h"

namespace WebCore {

class Color;

template<typename ColorType> inline double relativeLuminance(const ColorType& color)
{
    // https://en.wikipedia.org/wiki/Relative_luminance

    // FIXME: This can be optimized a bit by observing that in some cases the conversion
    // to XYZA<float, WhitePoint::D65> in its entirety is unnecessary to get just the Y
    // component. For instance, for SRGBA<float>, this could be done as:
    //
    //     convertColor<LinearSRGBA<float>>(color) * LinearSRGBA<float>::linearToXYZ.row(1)
    //
    // (for a hypothetical row() function on ColorMatrix). We would probably want to implement
    // this in ColorConversion.h as a sibling function to convertColor which can get a channel
    // of a color in another space in this kind of optimal way.

    return convertColor<XYZA<float, WhitePoint::D65>>(color).resolved().y;
}

inline double contrastRatio(double relativeLuminanceA, double relativeLuminanceB)
{
    // Uses the WCAG 2.0 definition of contrast ratio.
    // https://www.w3.org/TR/WCAG20/#contrast-ratiodef
    auto lighterLuminance = relativeLuminanceA;
    auto darkerLuminance = relativeLuminanceB;

    if (lighterLuminance < darkerLuminance)
        std::swap(lighterLuminance, darkerLuminance);

    return (lighterLuminance + 0.05) / (darkerLuminance + 0.05);
}

template<typename ColorTypeA, typename ColorTypeB> inline double contrastRatio(const ColorTypeA& colorA, const ColorTypeB& colorB)
{
    return contrastRatio(relativeLuminance(colorA), relativeLuminance(colorB));
}

double relativeLuminance(const Color&);
double contrastRatio(const Color&, const Color&);

}
