/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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
#include "HitTestLocation.h"

namespace WebCore {

HitTestLocation::HitTestLocation() = default;

HitTestLocation::HitTestLocation(const LayoutPoint& point)
    : m_point(point)
    , m_boundingBox(LayoutRect { flooredIntPoint(point), LayoutSize { 1, 1 } })
    , m_transformedPoint(point)
    , m_transformedRect(m_boundingBox)
{
}

HitTestLocation::HitTestLocation(const FloatPoint& point, const FloatQuad& quad)
    : m_point { flooredLayoutPoint(point) }
    , m_boundingBox { quad.enclosingBoundingBox() }
    , m_transformedPoint { point }
    , m_transformedRect { quad }
    , m_isRectBased { true }
    , m_isRectilinear { quad.isRectilinear() }
{
}

HitTestLocation::HitTestLocation(const LayoutRect& rect)
    : m_point { rect.center() } // Rounded to an integer point; it may not be the exact center.
    , m_boundingBox { rect }
    , m_transformedPoint { rect.center() }
    , m_transformedRect { FloatQuad { m_boundingBox } }
    , m_isRectBased { true }
{
}

HitTestLocation::HitTestLocation(const HitTestLocation& other, const LayoutSize& offset)
    : m_point(other.m_point)
    , m_boundingBox(other.m_boundingBox)
    , m_transformedPoint(other.m_transformedPoint)
    , m_transformedRect(other.m_transformedRect)
    , m_isRectBased(other.m_isRectBased)
    , m_isRectilinear(other.m_isRectilinear)
{
    move(offset);
}

HitTestLocation::HitTestLocation(const HitTestLocation& other)
    : m_point(other.m_point)
    , m_boundingBox(other.m_boundingBox)
    , m_transformedPoint(other.m_transformedPoint)
    , m_transformedRect(other.m_transformedRect)
    , m_isRectBased(other.m_isRectBased)
    , m_isRectilinear(other.m_isRectilinear)
{
}

HitTestLocation::~HitTestLocation() = default;

HitTestLocation& HitTestLocation::operator=(const HitTestLocation& other)
{
    m_point = other.m_point;
    m_boundingBox = other.m_boundingBox;
    m_transformedPoint = other.m_transformedPoint;
    m_transformedRect = other.m_transformedRect;
    m_isRectBased = other.m_isRectBased;
    m_isRectilinear = other.m_isRectilinear;

    return *this;
}

void HitTestLocation::move(const LayoutSize& offset)
{
    m_point.move(offset);
    m_transformedPoint.move(offset);
    m_transformedRect.move(offset);
    m_boundingBox = enclosingIntRect(m_transformedRect.boundingBox());
}

template<typename RectType>
bool HitTestLocation::intersectsRect(const RectType& rect) const
{
    // FIXME: When the hit test is not rect based we should use rect.contains(m_point).
    // That does change some corner case tests though.

    // First check if rect even intersects our bounding box.
    if (!rect.intersects(m_boundingBox))
        return false;

    // If the transformed rect is rectilinear the bounding box intersection was accurate.
    if (m_isRectilinear)
        return true;

    // If rect fully contains our bounding box, we are also sure of an intersection.
    if (rect.contains(m_boundingBox))
        return true;

    // Otherwise we need to do a slower quad based intersection test.
    return m_transformedRect.intersectsRect(rect);
}

bool HitTestLocation::intersects(const LayoutRect& rect) const
{
    return intersectsRect(rect);
}

bool HitTestLocation::intersects(const FloatRect& rect) const
{
    return intersectsRect(rect);
}

bool HitTestLocation::intersects(const RoundedRect& rect) const
{
    return rect.intersectsQuad(m_transformedRect);
}

} // namespace WebCore
