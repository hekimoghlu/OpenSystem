/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#include "LayoutUnit.h"

namespace WebCore {

class CollapsedBorderValue {
public:
    CollapsedBorderValue()
        : m_style(static_cast<unsigned>(BorderStyle::None))
        , m_precedence(static_cast<unsigned>(BorderPrecedence::Off))
        , m_transparent(false)
    {
    }

    CollapsedBorderValue(const BorderValue& border, const Color& color, BorderPrecedence precedence)
        : m_width(LayoutUnit(border.nonZero() ? border.width() : 0))
        , m_color(color)
        , m_style(static_cast<unsigned>(border.style()))
        , m_precedence(static_cast<unsigned>(precedence))
        , m_transparent(border.isTransparent())
    {
    }

    LayoutUnit width() const { return style() > BorderStyle::Hidden ? m_width : 0_lu; }
    BorderStyle style() const { return static_cast<BorderStyle>(m_style); }
    bool exists() const { return precedence() != BorderPrecedence::Off; }
    const Color& color() const { return m_color; }
    bool isTransparent() const { return m_transparent; }
    BorderPrecedence precedence() const { return static_cast<BorderPrecedence>(m_precedence); }

    bool isSameIgnoringColor(const CollapsedBorderValue& o) const
    {
        return width() == o.width() && style() == o.style() && precedence() == o.precedence();
    }

    static LayoutUnit adjustedCollapsedBorderWidth(float borderWidth, float deviceScaleFactor, bool roundUp);

private:
    LayoutUnit m_width;
    Color m_color;
    unsigned m_style : 4; // BorderStyle
    unsigned m_precedence : 3; // BorderPrecedence
    unsigned m_transparent : 1;
};

inline LayoutUnit CollapsedBorderValue::adjustedCollapsedBorderWidth(float borderWidth, float deviceScaleFactor, bool roundUp)
{
    float halfCollapsedBorderWidth = (borderWidth + (roundUp ? (1 / deviceScaleFactor) : 0)) / 2;
    return LayoutUnit(floorToDevicePixel(halfCollapsedBorderWidth, deviceScaleFactor));
}

} // namespace WebCore
