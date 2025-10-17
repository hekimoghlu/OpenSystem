/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include "FELightingSoftwareApplier.h"

namespace WebCore {

inline IntSize FELightingSoftwareApplier::LightingData::topLeftNormal(int offset) const
{
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int right = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    offset += widthMultipliedByPixelSize;
    int bottom = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int bottomRight = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    return {
        -2 * center + 2 * right - bottom + bottomRight,
        -2 * center - right + 2 * bottom + bottomRight
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::topRowNormal(int offset) const
{
    int left = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int right = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    offset += widthMultipliedByPixelSize;
    int bottomLeft = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int bottom = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int bottomRight = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    return {
        -2 * left + 2 * right - bottomLeft + bottomRight,
        -left - 2 * center - right + bottomLeft + 2 * bottom + bottomRight
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::topRightNormal(int offset) const
{
    int left = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    offset += widthMultipliedByPixelSize;
    int bottomLeft = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int bottom = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    return {
        -2 * left + 2 * center - bottomLeft + bottom,
        -left - 2 * center + bottomLeft + 2 * bottom
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::leftColumnNormal(int offset) const
{
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int right = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    offset -= widthMultipliedByPixelSize;
    int top = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int topRight = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    offset += 2 * widthMultipliedByPixelSize;
    int bottom = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int bottomRight = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    return {
        -top + topRight - 2 * center + 2 * right - bottom + bottomRight,
        -2 * top - topRight + 2 * bottom + bottomRight
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::interiorNormal(int offset, AlphaWindow& alphaWindow) const
{
    int rightAlphaOffset = offset + cPixelSize + cAlphaChannelOffset;

    int right = static_cast<int>(pixels->item(rightAlphaOffset));
    int topRight = static_cast<int>(pixels->item(rightAlphaOffset - widthMultipliedByPixelSize));
    int bottomRight = static_cast<int>(pixels->item(rightAlphaOffset + widthMultipliedByPixelSize));

    int left = alphaWindow.left();
    int topLeft = alphaWindow.topLeft();
    int top = alphaWindow.top();

    int bottomLeft = alphaWindow.bottomLeft();
    int bottom = alphaWindow.bottom();

    // The alphaWindow has been shifted, and here we fill in the right column.
    alphaWindow.alpha[0][2] = topRight;
    alphaWindow.alpha[1][2] = right;
    alphaWindow.alpha[2][2] = bottomRight;

    // Check that the alphaWindow is working with some spot-checks.
    ASSERT(alphaWindow.topLeft() == pixels->item(offset - cPixelSize - widthMultipliedByPixelSize + cAlphaChannelOffset)); // topLeft
    ASSERT(alphaWindow.top() == pixels->item(offset - widthMultipliedByPixelSize + cAlphaChannelOffset)); // top

    return {
        -topLeft + topRight - 2 * left + 2 * right - bottomLeft + bottomRight,
        -topLeft - 2 * top - topRight + bottomLeft + 2 * bottom + bottomRight
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::rightColumnNormal(int offset) const
{
    int left = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    offset -= widthMultipliedByPixelSize;
    int topLeft = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int top = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    offset += 2 * widthMultipliedByPixelSize;
    int bottomLeft = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int bottom = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    return {
        -topLeft + top - 2 * left + 2 * center - bottomLeft + bottom,
        -topLeft - 2 * top + bottomLeft + 2 * bottom
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::bottomLeftNormal(int offset) const
{
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int right = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    offset -= widthMultipliedByPixelSize;
    int top = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int topRight = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    return {
        -top + topRight - 2 * center + 2 * right,
        -2 * top - topRight + 2 * center + right
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::bottomRowNormal(int offset) const
{
    int left = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int right = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    offset -= widthMultipliedByPixelSize;
    int topLeft = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int top = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    int topRight = static_cast<int>(pixels->item(offset + cPixelSize + cAlphaChannelOffset));
    return {
        -topLeft + topRight - 2 * left + 2 * right,
        -topLeft - 2 * top - topRight + left + 2 * center + right
    };
}

inline IntSize FELightingSoftwareApplier::LightingData::bottomRightNormal(int offset) const
{
    int left = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int center = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    offset -= widthMultipliedByPixelSize;
    int topLeft = static_cast<int>(pixels->item(offset - cPixelSize + cAlphaChannelOffset));
    int top = static_cast<int>(pixels->item(offset + cAlphaChannelOffset));
    return {
        -topLeft + top - 2 * left + 2 * center,
        -topLeft - 2 * top + left + 2 * center
    };
}

} // namespace WebCore
