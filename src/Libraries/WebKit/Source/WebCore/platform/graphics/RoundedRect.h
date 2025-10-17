/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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

#include "LayoutRect.h"

namespace WebCore {

class FloatQuad;
class FloatRoundedRect;

class RoundedRectRadii {
public:
    RoundedRectRadii() = default;
    RoundedRectRadii(const LayoutSize& topLeft, const LayoutSize& topRight, const LayoutSize& bottomLeft, const LayoutSize& bottomRight)
        : m_topLeft(topLeft)
        , m_topRight(topRight)
        , m_bottomLeft(bottomLeft)
        , m_bottomRight(bottomRight)
    {
    }

    void setTopLeft(const LayoutSize& size) { m_topLeft = size; }
    void setTopRight(const LayoutSize& size) { m_topRight = size; }
    void setBottomLeft(const LayoutSize& size) { m_bottomLeft = size; }
    void setBottomRight(const LayoutSize& size) { m_bottomRight = size; }
    const LayoutSize& topLeft() const { return m_topLeft; }
    const LayoutSize& topRight() const { return m_topRight; }
    const LayoutSize& bottomLeft() const { return m_bottomLeft; }
    const LayoutSize& bottomRight() const { return m_bottomRight; }
    void setRadiiForEdges(const RoundedRectRadii&, RectEdges<bool> includeEdges);

    bool isZero() const;

    bool areRenderableInRect(const LayoutRect&) const;
    void makeRenderableInRect(const LayoutRect&);

    void scale(float factor);
    void expand(LayoutUnit topWidth, LayoutUnit bottomWidth, LayoutUnit leftWidth, LayoutUnit rightWidth);
    void expand(LayoutUnit size) { expand(size, size, size, size); }
    void shrink(LayoutUnit topWidth, LayoutUnit bottomWidth, LayoutUnit leftWidth, LayoutUnit rightWidth) { expand(-topWidth, -bottomWidth, -leftWidth, -rightWidth); }
    void shrink(LayoutUnit size) { shrink(size, size, size, size); }

    RoundedRectRadii transposedRadii() const { return { m_topLeft.transposedSize(), m_topRight.transposedSize(), m_bottomLeft.transposedSize(), m_bottomRight.transposedSize() }; }

    LayoutUnit minimumRadius() const { return std::min({ m_topLeft.minDimension(), m_topRight.minDimension(), m_bottomLeft.minDimension(), m_bottomRight.minDimension() }); }
    LayoutUnit maximumRadius() const { return std::max({ m_topLeft.minDimension(), m_topRight.minDimension(), m_bottomLeft.minDimension(), m_bottomRight.minDimension() }); }

    friend bool operator==(const RoundedRectRadii&, const RoundedRectRadii&) = default;

private:
    LayoutSize m_topLeft;
    LayoutSize m_topRight;
    LayoutSize m_bottomLeft;
    LayoutSize m_bottomRight;
};

class RoundedRect {
public:
    using Radii = RoundedRectRadii;

    WEBCORE_EXPORT explicit RoundedRect(const LayoutRect&, const Radii& = Radii());
    RoundedRect(LayoutUnit, LayoutUnit, LayoutUnit width, LayoutUnit height);
    WEBCORE_EXPORT RoundedRect(const LayoutRect&, const LayoutSize& topLeft, const LayoutSize& topRight, const LayoutSize& bottomLeft, const LayoutSize& bottomRight);

    const LayoutRect& rect() const { return m_rect; }
    const Radii& radii() const { return m_radii; }
    bool isRounded() const { return !m_radii.isZero(); }
    bool isEmpty() const { return m_rect.isEmpty(); }

    void setRect(const LayoutRect& rect) { m_rect = rect; }
    void setRadii(const Radii& radii) { m_radii = radii; }
    void setRadiiForEdges(const Radii& radii, RectEdges<bool> includeEdges) { m_radii.setRadiiForEdges(radii, includeEdges); }

    void move(const LayoutSize& size) { m_rect.move(size); }
    void moveBy(const LayoutPoint& offset) { m_rect.moveBy(offset); }
    void inflate(LayoutUnit size) { m_rect.inflate(size);  }
    void inflateWithRadii(LayoutUnit size);
    void expandRadii(LayoutUnit size) { m_radii.expand(size); }
    void shrinkRadii(LayoutUnit size) { m_radii.shrink(size); }

    bool isRenderable() const;
    void adjustRadii();

    // Tests whether the quad intersects any part of this rounded rectangle.
    // This only works for convex quads.
    bool intersectsQuad(const FloatQuad&) const;
    WEBCORE_EXPORT bool contains(const LayoutRect&) const;

    FloatRoundedRect pixelSnappedRoundedRectForPainting(float deviceScaleFactor) const;

    RoundedRect transposedRect() const { return RoundedRect(m_rect.transposedRect(), m_radii.transposedRadii()); }

    friend bool operator==(const RoundedRect&, const RoundedRect&) = default;

private:
    LayoutRect m_rect;
    Radii m_radii;
};

WTF::TextStream& operator<<(WTF::TextStream&, const RoundedRect&);

} // namespace WebCore
