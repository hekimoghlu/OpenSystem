/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

#include "Color.h"
#include "LayoutUnit.h"
#include "RectEdges.h"
#include "RenderObjectEnums.h"
#include "RenderStyleConstants.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class RenderStyle;

class BorderEdge {
public:
    BorderEdge() = default;
    BorderEdge(float edgeWidth, Color edgeColor, BorderStyle edgeStyle, bool edgeIsTransparent, bool edgeIsPresent, float devicePixelRatio);

    BorderStyle style() const { return m_style; }
    const Color& color() const { return m_color; }
    bool isTransparent() const { return m_isTransparent; }
    bool isPresent() const { return m_isPresent; }

    inline bool hasVisibleColorAndStyle() const { return m_style > BorderStyle::Hidden && !m_isTransparent; }
    inline bool shouldRender() const { return m_isPresent && widthForPainting() && hasVisibleColorAndStyle(); }
    inline bool presentButInvisible() const { return widthForPainting() && !hasVisibleColorAndStyle(); }
    inline float widthForPainting() const { return m_isPresent ?  m_flooredToDevicePixelWidth : 0; }
    void getDoubleBorderStripeWidths(LayoutUnit& outerWidth, LayoutUnit& innerWidth) const;
    bool obscuresBackgroundEdge(float scale) const;
    bool obscuresBackground() const;

private:
    inline float borderWidthInDevicePixel(int logicalPixels) const { return LayoutUnit(logicalPixels / m_devicePixelRatio).toFloat(); }

    Color m_color;
    LayoutUnit m_width;
    float m_flooredToDevicePixelWidth { 0 };
    float m_devicePixelRatio { 1 };
    BorderStyle m_style { BorderStyle::Hidden };
    bool m_isTransparent { false };
    bool m_isPresent { false };
};

using BorderEdges = RectEdges<BorderEdge>;
BorderEdges borderEdges(const RenderStyle&, float deviceScaleFactor, RectEdges<bool> closedEdges = { true }, bool setColorsToBlack = false);
BorderEdges borderEdgesForOutline(const RenderStyle&, float deviceScaleFactor);

inline bool edgesShareColor(const BorderEdge& firstEdge, const BorderEdge& secondEdge) { return firstEdge.color() == secondEdge.color(); }
inline BoxSideFlag edgeFlagForSide(BoxSide side) { return static_cast<BoxSideFlag>(1 << static_cast<unsigned>(side)); }
inline bool includesEdge(OptionSet<BoxSideFlag> flags, BoxSide side) { return flags.contains(edgeFlagForSide(side)); }

inline bool includesAdjacentEdges(OptionSet<BoxSideFlag> flags)
{
    // The set includes adjacent edges if and only if it contains at least one horizontal and one vertical edge.
    return flags.containsAny({ BoxSideFlag::Top, BoxSideFlag::Bottom })
        && flags.containsAny({ BoxSideFlag::Left, BoxSideFlag::Right });
}

} // namespace WebCore
