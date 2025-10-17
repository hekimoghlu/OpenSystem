/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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

#include "BorderValue.h"
#include "LengthSize.h"
#include "NinePieceImage.h"

namespace WebCore {

class OutlineValue;

struct BorderDataRadii {
    LengthSize topLeft { LengthType::Fixed, LengthType::Fixed };
    LengthSize topRight { LengthType::Fixed, LengthType::Fixed };
    LengthSize bottomLeft { LengthType::Fixed, LengthType::Fixed };
    LengthSize bottomRight { LengthType::Fixed, LengthType::Fixed };

    friend bool operator==(const BorderDataRadii&, const BorderDataRadii&) = default;
};

class BorderData {
friend class RenderStyle;
public:
    using Radii = BorderDataRadii;

    bool hasBorder() const
    {
        return m_left.nonZero() || m_right.nonZero() || m_top.nonZero() || m_bottom.nonZero();
    }

    bool hasVisibleBorder() const
    {
        return m_left.isVisible() || m_right.isVisible() || m_top.isVisible() || m_bottom.isVisible();
    }

    bool hasBorderImage() const
    {
        return m_image.hasImage();
    }

    bool hasBorderRadius() const
    {
        return !m_radii.topLeft.isEmpty()
            || !m_radii.topRight.isEmpty()
            || !m_radii.bottomLeft.isEmpty()
            || !m_radii.bottomRight.isEmpty();
    }

    float borderLeftWidth() const
    {
        if (m_left.style() == BorderStyle::None || m_left.style() == BorderStyle::Hidden)
            return 0;
        if (m_image.overridesBorderWidths() && m_image.borderSlices().left().isFixed())
            return m_image.borderSlices().left().value();
        return m_left.width();
    }

    float borderRightWidth() const
    {
        if (m_right.style() == BorderStyle::None || m_right.style() == BorderStyle::Hidden)
            return 0;
        if (m_image.overridesBorderWidths() && m_image.borderSlices().right().isFixed())
            return m_image.borderSlices().right().value();
        return m_right.width();
    }

    float borderTopWidth() const
    {
        if (m_top.style() == BorderStyle::None || m_top.style() == BorderStyle::Hidden)
            return 0;
        if (m_image.overridesBorderWidths() && m_image.borderSlices().top().isFixed())
            return m_image.borderSlices().top().value();
        return m_top.width();
    }

    float borderBottomWidth() const
    {
        if (m_bottom.style() == BorderStyle::None || m_bottom.style() == BorderStyle::Hidden)
            return 0;
        if (m_image.overridesBorderWidths() && m_image.borderSlices().bottom().isFixed())
            return m_image.borderSlices().bottom().value();
        return m_bottom.width();
    }

    FloatBoxExtent borderWidth() const
    {
        return FloatBoxExtent(borderTopWidth(), borderRightWidth(), borderBottomWidth(), borderLeftWidth());
    }

    bool isEquivalentForPainting(const BorderData& other, bool currentColorDiffers) const;

    friend bool operator==(const BorderData&, const BorderData&) = default;

    const BorderValue& left() const { return m_left; }
    const BorderValue& right() const { return m_right; }
    const BorderValue& top() const { return m_top; }
    const BorderValue& bottom() const { return m_bottom; }

    const NinePieceImage& image() const { return m_image; }

    const LengthSize& topLeftRadius() const { return m_radii.topLeft; }
    const LengthSize& topRightRadius() const { return m_radii.topRight; }
    const LengthSize& bottomLeftRadius() const { return m_radii.bottomLeft; }
    const LengthSize& bottomRightRadius() const { return m_radii.bottomRight; }

    void dump(TextStream&, DumpStyleValues = DumpStyleValues::All) const;

private:
    BorderValue m_left;
    BorderValue m_right;
    BorderValue m_top;
    BorderValue m_bottom;

    NinePieceImage m_image;

    Radii m_radii;
};

WTF::TextStream& operator<<(WTF::TextStream&, const BorderValue&);
WTF::TextStream& operator<<(WTF::TextStream&, const OutlineValue&);
WTF::TextStream& operator<<(WTF::TextStream&, const BorderData&);

} // namespace WebCore
