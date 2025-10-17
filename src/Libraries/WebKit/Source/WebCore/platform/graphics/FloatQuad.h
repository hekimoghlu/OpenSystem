/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "FloatRect.h"
#include "IntRect.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

// FIXME: Seems like this would be better as a struct.

// A FloatQuad is a collection of 4 points, often representing the result of
// mapping a rectangle through transforms. When initialized from a rect, the
// points are in clockwise order from top left.
class FloatQuad {
    WTF_MAKE_TZONE_ALLOCATED(FloatQuad);
public:
    FloatQuad()
    {
    }

    FloatQuad(const FloatPoint& p1, const FloatPoint& p2, const FloatPoint& p3, const FloatPoint& p4)
        : m_p1(p1)
        , m_p2(p2)
        , m_p3(p3)
        , m_p4(p4)
    {
    }

    FloatQuad(const FloatRect& inRect)
        : m_p1(inRect.location())
        , m_p2(inRect.maxX(), inRect.y())
        , m_p3(inRect.maxX(), inRect.maxY())
        , m_p4(inRect.x(), inRect.maxY())
    {
    }

    const FloatPoint& p1() const { return m_p1; }
    const FloatPoint& p2() const { return m_p2; }
    const FloatPoint& p3() const { return m_p3; }
    const FloatPoint& p4() const { return m_p4; }

    void setP1(const FloatPoint& p) { m_p1 = p; }
    void setP2(const FloatPoint& p) { m_p2 = p; }
    void setP3(const FloatPoint& p) { m_p3 = p; }
    void setP4(const FloatPoint& p) { m_p4 = p; }

    WEBCORE_EXPORT bool isEmpty() const;

    // This method will not identify "slanted" empty quads.
    bool boundingBoxIsEmpty() const { return boundingBox().isEmpty(); }

    // Tests whether this quad can be losslessly represented by a FloatRect,
    // that is, if two edges are parallel to the x-axis and the other two
    // are parallel to the y-axis. If this method returns true, the
    // corresponding FloatRect can be retrieved with boundingBox().
    WEBCORE_EXPORT bool isRectilinear() const;

    // Tests whether the given point is inside, or on an edge or corner of this quad.
    WEBCORE_EXPORT bool containsPoint(const FloatPoint&) const;

    // Tests whether the four corners of other are inside, or coincident with the sides of this quad.
    // Note that this only works for convex quads, but that includes all quads that originate
    // from transformed rects.
    WEBCORE_EXPORT bool containsQuad(const FloatQuad&) const;

    // Tests whether any part of the rectangle intersects with this quad.
    // This only works for convex quads.
    bool intersectsRect(const FloatRect&) const;

    // Test whether any part of the circle/ellipse intersects with this quad.
    // Note that these two functions only work for convex quads.
    bool intersectsCircle(const FloatPoint& center, float radius) const;
    bool intersectsEllipse(const FloatPoint& center, const FloatSize& radii) const;

    // The center of the quad. If the quad is the result of a affine-transformed rectangle this is the same as the original center transformed.
    FloatPoint center() const
    {
        return FloatPoint((m_p1.x() + m_p2.x() + m_p3.x() + m_p4.x()) / 4.0,
                          (m_p1.y() + m_p2.y() + m_p3.y() + m_p4.y()) / 4.0);
    }

    WEBCORE_EXPORT FloatRect boundingBox() const;
    IntRect enclosingBoundingBox() const
    {
        return enclosingIntRect(boundingBox());
    }

    void move(const FloatSize& offset)
    {
        m_p1 += offset;
        m_p2 += offset;
        m_p3 += offset;
        m_p4 += offset;
    }

    void move(float dx, float dy)
    {
        m_p1.move(dx, dy);
        m_p2.move(dx, dy);
        m_p3.move(dx, dy);
        m_p4.move(dx, dy);
    }
    
    void scale(float s)
    {
        scale(s, s);
    }

    void scale(float dx, float dy)
    {
        m_p1.scale(dx, dy);
        m_p2.scale(dx, dy);
        m_p3.scale(dx, dy);
        m_p4.scale(dx, dy);
    }

    // Tests whether points are in clock-wise, or counter clock-wise order.
    // Note that output is undefined when all points are colinear.
    bool isCounterclockwise() const;

    friend bool operator==(const FloatQuad&, const FloatQuad&) = default;

private:
    FloatPoint m_p1;
    FloatPoint m_p2;
    FloatPoint m_p3;
    FloatPoint m_p4;
};

inline FloatQuad& operator+=(FloatQuad& a, const FloatSize& b)
{
    a.move(b);
    return a;
}

inline FloatQuad& operator-=(FloatQuad& a, const FloatSize& b)
{
    a.move(-b.width(), -b.height());
    return a;
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FloatQuad&);

Vector<FloatRect> boundingBoxes(const Vector<FloatQuad>&);
WEBCORE_EXPORT FloatRect unitedBoundingBoxes(const Vector<FloatQuad>&);

} // namespace WebCore
