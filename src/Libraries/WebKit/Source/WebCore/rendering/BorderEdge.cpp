/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#include "BorderEdge.h"

#include "Color.h"
#include "LayoutUnit.h"
#include "RenderObject.h"
#include "RenderStyleInlines.h"

namespace WebCore {

BorderEdge::BorderEdge(float edgeWidth, Color edgeColor, BorderStyle edgeStyle, bool edgeIsTransparent, bool edgeIsPresent, float devicePixelRatio)
    : m_color(edgeColor)
    , m_width(edgeWidth)
    , m_devicePixelRatio(devicePixelRatio)
    , m_style(edgeStyle)
    , m_isTransparent(edgeIsTransparent)
    , m_isPresent(edgeIsPresent)
{
    if (edgeStyle == BorderStyle::Double && edgeWidth  < borderWidthInDevicePixel(3))
        m_style = BorderStyle::Solid;
    m_flooredToDevicePixelWidth = floorf(edgeWidth * devicePixelRatio) / devicePixelRatio;
}

BorderEdges borderEdges(const RenderStyle& style, float deviceScaleFactor, RectEdges<bool> closedEdges, bool setColorsToBlack)
{
    auto constructBorderEdge = [&](float width, CSSPropertyID borderColorProperty, BorderStyle borderStyle, bool isTransparent, bool isPresent) {
        auto color = setColorsToBlack ? Color::black : style.visitedDependentColorWithColorFilter(borderColorProperty);
        return BorderEdge(width, color, borderStyle, !setColorsToBlack && isTransparent, isPresent, deviceScaleFactor);
    };

    return {
        constructBorderEdge(style.borderTopWidth(), CSSPropertyBorderTopColor, style.borderTopStyle(), style.borderTopIsTransparent(), closedEdges.top()),
        constructBorderEdge(style.borderRightWidth(), CSSPropertyBorderRightColor, style.borderRightStyle(), style.borderRightIsTransparent(), closedEdges.right()),
        constructBorderEdge(style.borderBottomWidth(), CSSPropertyBorderBottomColor, style.borderBottomStyle(), style.borderBottomIsTransparent(), closedEdges.bottom()),
        constructBorderEdge(style.borderLeftWidth(), CSSPropertyBorderLeftColor, style.borderLeftStyle(), style.borderLeftIsTransparent(), closedEdges.left())
    };
}

BorderEdges borderEdgesForOutline(const RenderStyle& style, float deviceScaleFactor)
{
    auto color = style.visitedDependentColorWithColorFilter(CSSPropertyOutlineColor);
    auto isTransparent = color.isValid() && !color.isVisible();
    auto size = style.outlineWidth();
    auto outlineStyle = style.outlineStyle();
    return {
        BorderEdge { size, color, outlineStyle, isTransparent, true, deviceScaleFactor },
        BorderEdge { size, color, outlineStyle, isTransparent, true, deviceScaleFactor },
        BorderEdge { size, color, outlineStyle, isTransparent, true, deviceScaleFactor },
        BorderEdge { size, color, outlineStyle, isTransparent, true, deviceScaleFactor } 
    };
}

bool BorderEdge::obscuresBackgroundEdge(float scale) const
{
    if (!m_isPresent || m_isTransparent || (m_width * scale) < borderWidthInDevicePixel(2) || !m_color.isOpaque() || m_style == BorderStyle::Hidden)
        return false;

    if (m_style == BorderStyle::Dotted || m_style == BorderStyle::Dashed)
        return false;

    if (m_style == BorderStyle::Double)
        return m_width >= scale * borderWidthInDevicePixel(5); // The outer band needs to be >= 2px wide at unit scale.

    return true;
}

bool BorderEdge::obscuresBackground() const
{
    if (!m_isPresent || m_isTransparent || !m_color.isOpaque() || m_style == BorderStyle::Hidden)
        return false;

    if (m_style == BorderStyle::Dotted || m_style == BorderStyle::Dashed || m_style == BorderStyle::Double)
        return false;

    return true;
}

void BorderEdge::getDoubleBorderStripeWidths(LayoutUnit& outerWidth, LayoutUnit& innerWidth) const
{
    LayoutUnit fullWidth { widthForPainting() };
    innerWidth = ceilToDevicePixel(fullWidth * 2 / 3, m_devicePixelRatio);
    outerWidth = floorToDevicePixel(fullWidth / 3, m_devicePixelRatio);
}

} // namespace WebCore
