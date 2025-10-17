/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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

#include "FloatQuad.h"
#include "RoundedRect.h"

namespace WebCore {

class HitTestLocation {
public:
    WEBCORE_EXPORT HitTestLocation();
    HitTestLocation(const LayoutPoint&);
    HitTestLocation(const FloatPoint&, const FloatQuad&);

    HitTestLocation(const LayoutRect&);

    // Make a copy the HitTestLocation in a new region by applying given offset to internal point and area.
    HitTestLocation(const HitTestLocation&, const LayoutSize& offset);
    WEBCORE_EXPORT HitTestLocation(const HitTestLocation&);
    WEBCORE_EXPORT ~HitTestLocation();
    HitTestLocation& operator=(const HitTestLocation&);

    const LayoutPoint& point() const { return m_point; }
    IntPoint roundedPoint() const { return roundedIntPoint(m_point); }

    // Rect-based hit test related methods.
    bool isRectBasedTest() const { return m_isRectBased; }
    bool isRectilinear() const { return m_isRectilinear; }
    LayoutRect boundingBox() const { return m_boundingBox; }
    
    int topPadding() const { return roundedPoint().y() - m_boundingBox.y(); }
    int rightPadding() const { return m_boundingBox.maxX() - roundedPoint().x() - 1; }
    int bottomPadding() const { return m_boundingBox.maxY() - roundedPoint().y() - 1; }
    int leftPadding() const { return roundedPoint().x() - m_boundingBox.x(); }

    WEBCORE_EXPORT bool intersects(const LayoutRect&) const;
    bool intersects(const FloatRect&) const;
    bool intersects(const RoundedRect&) const;

    const FloatPoint& transformedPoint() const { return m_transformedPoint; }
    const FloatQuad& transformedRect() const { return m_transformedRect; }

private:
    template<typename RectType> bool intersectsRect(const RectType&) const;

    void move(const LayoutSize&);

    // These are the cached forms of the more accurate point and rect below.
    LayoutPoint m_point;
    LayoutRect m_boundingBox;

    FloatPoint m_transformedPoint;
    FloatQuad m_transformedRect;

    bool m_isRectBased { false };
    bool m_isRectilinear { true };
};

} // namespace WebCore
