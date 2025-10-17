/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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

#include "AffineTransform.h"
#include "FloatSize.h"
#include <stdint.h>

// X11 headers define a bunch of macros with common terms, interfering with WebCore and WTF enum values.
// As a workaround, we explicitly undef them here.
#if defined(None)
#undef None
#endif

namespace WebCore {

struct ImageOrientation {
    enum class Orientation : uint8_t {
        FromImage         = 0, // Orientation from the image should be respected.

        // This range intentionally matches the orientation values from the EXIF spec.
        // See JEITA CP-3451, page 18. http://www.exif.org/Exif2-2.PDF
        OriginTopLeft     = 1, // default
        OriginTopRight    = 2, // mirror along y-axis
        OriginBottomRight = 3, // 180 degree rotation
        OriginBottomLeft  = 4, // mirror along the x-axis
        OriginLeftTop     = 5, // mirror along x-axis + 270 degree CW rotation
        OriginRightTop    = 6, // 90 degree CW rotation
        OriginRightBottom = 7, // mirror along x-axis + 90 degree CW rotation
        OriginLeftBottom  = 8, // 270 degree CW rotation

        None              = OriginTopLeft
    };

    constexpr ImageOrientation() = default;

    constexpr ImageOrientation(Orientation orientation)
        : m_orientation(orientation)
    {
    }
    
    ImageOrientation(int orientation)
    {
        RELEASE_ASSERT(isValidOrientation(orientation));
        m_orientation = static_cast<Orientation>(orientation);
    }

    static constexpr Orientation fromEXIFValue(int exifValue)
    {
        return isValidEXIFOrientation(exifValue) ? static_cast<Orientation>(exifValue) : Orientation::None;
    }

    constexpr operator int() const { return static_cast<int>(m_orientation); }
    friend constexpr bool operator==(const ImageOrientation&, const ImageOrientation&) = default;

    constexpr Orientation orientation() const { return m_orientation; }
    
    constexpr bool usesWidthAsHeight() const
    {
        ASSERT(m_orientation != Orientation::FromImage);
        // Values 5 through 8 all flip the width/height.
        switch (m_orientation) {
        case Orientation::OriginLeftTop:
        case Orientation::OriginRightTop:
        case Orientation::OriginRightBottom:
        case Orientation::OriginLeftBottom:
            return true;
        default:
            return false;
        }
    }

    constexpr AffineTransform transformFromDefault(const FloatSize& drawnSize) const
    {
        float w = drawnSize.width();
        float h = drawnSize.height();

        switch (m_orientation) {
        case Orientation::FromImage:
            ASSERT_NOT_REACHED();
            return AffineTransform();
        case Orientation::OriginTopLeft:
            return AffineTransform();
        case Orientation::OriginTopRight:
            return AffineTransform(-1,  0,  0,  1,  w, 0);
        case Orientation::OriginBottomRight:
            return AffineTransform(-1,  0,  0, -1,  w, h);
        case Orientation::OriginBottomLeft:
            return AffineTransform( 1,  0,  0, -1,  0, h);
        case Orientation::OriginLeftTop:
            return AffineTransform( 0,  1,  1,  0,  0, 0);
        case Orientation::OriginRightTop:
            return AffineTransform( 0,  1, -1,  0,  w, 0);
        case Orientation::OriginRightBottom:
            return AffineTransform( 0, -1, -1,  0,  w, h);
        case Orientation::OriginLeftBottom:
            return AffineTransform( 0, -1,  1,  0,  0, h);
        }

        ASSERT_NOT_REACHED();
        return AffineTransform();
    }

    constexpr ImageOrientation withFlippedY() const
    {
        ASSERT(m_orientation != Orientation::FromImage);

        switch (m_orientation) {
        case Orientation::FromImage:
            ASSERT_NOT_REACHED();
            return Orientation::None;
        case Orientation::OriginTopLeft:
            return Orientation::OriginBottomLeft;
        case Orientation::OriginTopRight:
            return Orientation::OriginBottomRight;
        case Orientation::OriginBottomRight:
            return Orientation::OriginTopRight;
        case Orientation::OriginBottomLeft:
            return Orientation::OriginTopLeft;
        case Orientation::OriginLeftTop:
            return Orientation::OriginLeftBottom;
        case Orientation::OriginRightTop:
            return Orientation::OriginRightBottom;
        case Orientation::OriginRightBottom:
            return Orientation::OriginRightTop;
        case Orientation::OriginLeftBottom:
            return Orientation::OriginLeftTop;
        }

        ASSERT_NOT_REACHED();
        return Orientation::None;
    }

private:
    static constexpr auto EXIFFirst = Orientation::OriginTopLeft;
    static constexpr auto EXIFLast = Orientation::OriginLeftBottom;
    static constexpr auto First = Orientation::FromImage;
    static constexpr auto Last = EXIFLast;

    static constexpr bool isValidOrientation(int orientation)
    {
        return orientation >= static_cast<int>(First) && orientation <= static_cast<int>(Last);
    }

    static constexpr bool isValidEXIFOrientation(int orientation)
    {
        return orientation >= static_cast<int>(EXIFFirst) && orientation <= static_cast<int>(EXIFLast);
    }

    Orientation m_orientation { Orientation::None };
};

TextStream& operator<<(TextStream&, ImageOrientation::Orientation);

} // namespace WebCore
