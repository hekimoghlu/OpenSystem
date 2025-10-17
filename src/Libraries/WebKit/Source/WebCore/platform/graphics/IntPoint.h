/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#include "IntSize.h"
#include <cmath>

#if USE(CG)
typedef struct CGPoint CGPoint;
#endif

#if !PLATFORM(IOS_FAMILY)
#if OS(DARWIN)
#ifdef NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES
typedef struct CGPoint NSPoint;
#else
typedef struct _NSPoint NSPoint;
#endif
#endif
#endif // !PLATFORM(IOS_FAMILY)

#if PLATFORM(WIN)
typedef struct tagPOINT POINT;
typedef struct tagPOINTS POINTS;
#endif

namespace WTF {
class TextStream;
}

namespace WebCore {

class FloatPoint;
class IntRect;

class IntPoint {
public:
    constexpr IntPoint() : m_x(0), m_y(0) { }
    constexpr IntPoint(int x, int y) : m_x(x), m_y(y) { }
    explicit IntPoint(const IntSize& size) : m_x(size.width()), m_y(size.height()) { }
    WEBCORE_EXPORT explicit IntPoint(const FloatPoint&); // don't do this implicitly since it's lossy

    static constexpr IntPoint zero() { return IntPoint(); }
    constexpr bool isZero() const { return !m_x && !m_y; }

    constexpr int x() const { return m_x; }
    constexpr int y() const { return m_y; }

    void setX(int x) { m_x = x; }
    void setY(int y) { m_y = y; }

    void move(const IntSize& s) { move(s.width(), s.height()); } 
    void moveBy(const IntPoint& offset) { move(offset.x(), offset.y()); }
    void move(int dx, int dy) { m_x += dx; m_y += dy; }
    void scale(float sx, float sy)
    {
        m_x = lroundf(static_cast<float>(m_x * sx));
        m_y = lroundf(static_cast<float>(m_y * sy));
    }

    void scale(float scale)
    {
        this->scale(scale, scale);
    }
    
    constexpr IntPoint expandedTo(const IntPoint& other) const
    {
        return {
            m_x > other.m_x ? m_x : other.m_x,
            m_y > other.m_y ? m_y : other.m_y
        };
    }

    constexpr IntPoint shrunkTo(const IntPoint& other) const
    {
        return {
            m_x < other.m_x ? m_x : other.m_x,
            m_y < other.m_y ? m_y : other.m_y
        };
    }

    WEBCORE_EXPORT IntPoint constrainedBetween(const IntPoint& min, const IntPoint& max) const;
    
    WEBCORE_EXPORT IntPoint constrainedWithin(const IntRect&) const;

    int distanceSquaredToPoint(const IntPoint&) const;

    void clampNegativeToZero()
    {
        *this = expandedTo(zero());
    }

    IntPoint transposedPoint() const
    {
        return IntPoint(m_y, m_x);
    }

    friend bool operator==(const IntPoint&, const IntPoint&) = default;

#if USE(CG)
    WEBCORE_EXPORT explicit IntPoint(const CGPoint&); // don't do this implicitly since it's lossy
    WEBCORE_EXPORT operator CGPoint() const;
#endif

#if !PLATFORM(IOS_FAMILY)
#if OS(DARWIN) && !defined(NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES)
    WEBCORE_EXPORT explicit IntPoint(const NSPoint&); // don't do this implicitly since it's lossy
    WEBCORE_EXPORT operator NSPoint() const;
#endif
#endif // !PLATFORM(IOS_FAMILY)

#if PLATFORM(WIN)
    WEBCORE_EXPORT IntPoint(const POINT&);
    WEBCORE_EXPORT operator POINT() const;
    WEBCORE_EXPORT IntPoint(const POINTS&);
    operator POINTS() const;
#endif

private:
    int m_x, m_y;
};

inline IntPoint& operator+=(IntPoint& a, const IntSize& b)
{
    a.move(b.width(), b.height());
    return a;
}

inline IntPoint& operator-=(IntPoint& a, const IntSize& b)
{
    a.move(-b.width(), -b.height());
    return a;
}

inline IntPoint operator+(const IntPoint& a, const IntSize& b)
{
    return IntPoint(a.x() + b.width(), a.y() + b.height());
}

inline IntPoint operator+(const IntPoint& a, const IntPoint& b)
{
    return IntPoint(a.x() + b.x(), a.y() + b.y());
}

inline IntSize operator-(const IntPoint& a, const IntPoint& b)
{
    return IntSize(a.x() - b.x(), a.y() - b.y());
}

inline IntPoint operator-(const IntPoint& a, const IntSize& b)
{
    return IntPoint(a.x() - b.width(), a.y() - b.height());
}

inline IntPoint operator-(const IntPoint& point)
{
    return IntPoint(-point.x(), -point.y());
}

inline IntSize toIntSize(const IntPoint& a)
{
    return IntSize(a.x(), a.y());
}

inline int IntPoint::distanceSquaredToPoint(const IntPoint& point) const
{
    return ((*this) - point).diagonalLengthSquared();
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const IntPoint&);

} // namespace WebCore

namespace WTF {
template<> struct DefaultHash<WebCore::IntPoint>;
template<> struct HashTraits<WebCore::IntPoint>;
}
