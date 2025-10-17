/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

#include "RectEdges.h"
#include "RenderStyleConstants.h"
#include "RoundedRect.h"

namespace WebCore {

class Color;
class GraphicsContext;
class FloatRect;
class FloatRoundedRect;
class Path;
class RenderStyle;

// BorderShape is used to fill and clip to the shape formed by the border and padding boxes with border-radius.
// In future, this may be a more complex shape than a rounded rect, so accessors that return rounded rects
// are deprecated.
class BorderShape {
public:
    static BorderShape shapeForBorderRect(const RenderStyle&, const LayoutRect& borderRect, RectEdges<bool> closedEdges = { true });
    // overrideBorderWidths describe custom insets from the border box, used instead of the border widths from the style.
    static BorderShape shapeForBorderRect(const RenderStyle&, const LayoutRect& borderRect, const RectEdges<LayoutUnit>& overrideBorderWidths, RectEdges<bool> closedEdges = { true });

    BorderShape(const LayoutRect& borderRect, const RectEdges<LayoutUnit>& borderWidths);
    BorderShape(const LayoutRect& borderRect, const RectEdges<LayoutUnit>& borderWidths, const RoundedRectRadii&);

    BorderShape(const BorderShape& other)
        : m_borderRect(other.m_borderRect)
        , m_borderWidths(other.m_borderWidths)
    {
    }

    LayoutRect borderRect() const { return m_borderRect.rect(); }

    RoundedRect deprecatedRoundedRect() const;
    RoundedRect deprecatedInnerRoundedRect() const;
    FloatRoundedRect deprecatedPixelSnappedRoundedRect(float deviceScaleFactor) const;
    FloatRoundedRect deprecatedPixelSnappedInnerRoundedRect(float deviceScaleFactor) const;

    // Returns true if the given rect is entirely inside the shape, without impinging on any of the corners.
    bool innerShapeContains(const LayoutRect&) const;
    bool outerShapeContains(const LayoutRect&) const;

    const RoundedRectRadii& radii() const { return m_borderRect.radii(); }
    void setRadii(const RoundedRectRadii& radii) { m_borderRect.setRadii(radii); }

    FloatRect snappedOuterRect(float deviceScaleFactor) const;
    FloatRect snappedInnerRect(float deviceScaleFactor) const;

    bool isRounded() const { return m_borderRect.isRounded(); }
    bool isEmpty() const { return m_borderRect.rect().isEmpty(); }

    void move(LayoutSize);

    // This will inflate the m_borderRect, and scale the radii up accordingly. Note that this changes the meaning of "inner shape" which will no longer correspond to the padding box.
    void inflate(LayoutUnit);

    Path pathForOuterShape(float deviceScaleFactor) const;
    Path pathForInnerShape(float deviceScaleFactor) const;

    Path pathForBorderArea(float deviceScaleFactor) const;

    void clipToOuterShape(GraphicsContext&, float deviceScaleFactor);
    void clipToInnerShape(GraphicsContext&, float deviceScaleFactor);

    void clipOutOuterShape(GraphicsContext&, float deviceScaleFactor);
    void clipOutInnerShape(GraphicsContext&, float deviceScaleFactor);

    void fillOuterShape(GraphicsContext&, const Color&, float deviceScaleFactor);
    void fillInnerShape(GraphicsContext&, const Color&, float deviceScaleFactor);

private:
    RoundedRect innerEdgeRoundedRect() const;
    LayoutRect innerEdgeRect() const;

    RoundedRect m_borderRect;
    RectEdges<LayoutUnit> m_borderWidths;
};

} // namespace WebCore
