/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

#include "FloatPoint.h"
#include "LayoutSize.h"

namespace WebCore {

class LayoutPoint {
public:
    constexpr LayoutPoint() = default;
    template<typename T, typename U> LayoutPoint(T x, U y) : m_x(x), m_y(y) { }
    LayoutPoint(const IntPoint& point) : m_x(point.x()), m_y(point.y()) { }
    explicit LayoutPoint(const FloatPoint& size) : m_x(size.x()), m_y(size.y()) { }
    explicit LayoutPoint(const LayoutSize& size) : m_x(size.width()), m_y(size.height()) { }

    static constexpr LayoutPoint zero() { return LayoutPoint(); }
    constexpr bool isZero() const { return !m_x && !m_y; }

    LayoutUnit x() const { return m_x; }
    LayoutUnit y() const { return m_y; }

    template<typename T> void setX(T x) { m_x = x; }
    template<typename T> void setY(T y) { m_y = y; }

    void move(const LayoutSize& s) { move(s.width(), s.height()); } 
    void moveBy(const LayoutPoint& offset) { move(offset.x(), offset.y()); }
    template<typename T, typename U> void move(T dx, U dy) { m_x += dx; m_y += dy; }
    
    friend bool operator==(const LayoutPoint&, const LayoutPoint&) = default;

    void scale(float s)
    {
        m_x *= s;
        m_y *= s;
    }
    
    void scale(float sx, float sy)
    {
        m_x *= sx;
        m_y *= sy;
    }

    LayoutPoint scaled(float s) const
    {
        return { m_x * s, m_y * s };
    }

    LayoutPoint scaled(float sx, float sy) const
    {
        return { m_x * sx, m_y * sy };
    }

    LayoutPoint constrainedBetween(const LayoutPoint& min, const LayoutPoint& max) const;

    LayoutPoint expandedTo(const LayoutPoint& other) const
    {
        return { std::max(m_x, other.m_x), std::max(m_y, other.m_y) };
    }

    LayoutPoint shrunkTo(const LayoutPoint& other) const
    {
        return { std::min(m_x, other.m_x), std::min(m_y, other.m_y) };
    }

    void clampNegativeToZero()
    {
        *this = expandedTo(zero());
    }

    LayoutPoint transposedPoint() const
    {
        return { m_y, m_x };
    }

    LayoutPoint fraction() const
    {
        return { m_x.fraction(), m_y.fraction() };
    }

    operator FloatPoint() const { return { m_x, m_y }; }

private:
    LayoutUnit m_x, m_y;
};

inline LayoutPoint& operator+=(LayoutPoint& a, const LayoutSize& b)
{
    a.move(b.width(), b.height());
    return a;
}

inline LayoutPoint& operator-=(LayoutPoint& a, const LayoutSize& b)
{
    a.move(-b.width(), -b.height());
    return a;
}

inline LayoutPoint operator+(const LayoutPoint& a, const LayoutSize& b)
{
    return LayoutPoint(a.x() + b.width(), a.y() + b.height());
}

inline LayoutPoint operator+(const LayoutPoint& a, const LayoutPoint& b)
{
    return LayoutPoint(a.x() + b.x(), a.y() + b.y());
}

inline LayoutSize operator-(const LayoutPoint& a, const LayoutPoint& b)
{
    return LayoutSize(a.x() - b.x(), a.y() - b.y());
}

inline LayoutPoint operator-(const LayoutPoint& a, const LayoutSize& b)
{
    return LayoutPoint(a.x() - b.width(), a.y() - b.height());
}

inline LayoutPoint operator-(const LayoutPoint& point)
{
    return LayoutPoint(-point.x(), -point.y());
}

inline LayoutPoint toLayoutPoint(const LayoutSize& size)
{
    return LayoutPoint(size.width(), size.height());
}

inline LayoutSize toLayoutSize(const LayoutPoint& point)
{
    return LayoutSize(point.x(), point.y());
}

inline IntPoint flooredIntPoint(const LayoutPoint& point)
{
    return IntPoint(point.x().floor(), point.y().floor());
}

inline IntPoint roundedIntPoint(const LayoutPoint& point)
{
    return IntPoint(point.x().round(), point.y().round());
}

inline IntPoint roundedIntPoint(const LayoutSize& size)
{
    return roundedIntPoint(toLayoutPoint(size));
}

inline IntPoint ceiledIntPoint(const LayoutPoint& point)
{
    return IntPoint(point.x().ceil(), point.y().ceil());
}

inline LayoutPoint flooredLayoutPoint(const FloatPoint& p)
{
    return LayoutPoint(LayoutUnit::fromFloatFloor(p.x()), LayoutUnit::fromFloatFloor(p.y()));
}

inline LayoutPoint ceiledLayoutPoint(const FloatPoint& p)
{
    return LayoutPoint(LayoutUnit::fromFloatCeil(p.x()), LayoutUnit::fromFloatCeil(p.y()));
}

inline IntSize snappedIntSize(const LayoutSize& size, const LayoutPoint& location)
{
    auto snap = [] (LayoutUnit a, LayoutUnit b) {
        LayoutUnit fraction = b.fraction();
        return roundToInt(fraction + a) - roundToInt(fraction);
    };
    return IntSize(snap(size.width(), location.x()), snap(size.height(), location.y()));
}

inline FloatPoint roundPointToDevicePixels(const LayoutPoint& point, float pixelSnappingFactor, bool directionalRoundingToRight = true, bool directionalRoundingToBottom = true)
{
    return FloatPoint(roundToDevicePixel(point.x(), pixelSnappingFactor, !directionalRoundingToRight), roundToDevicePixel(point.y(), pixelSnappingFactor, !directionalRoundingToBottom));
}

inline FloatPoint floorPointToDevicePixels(const LayoutPoint& point, float pixelSnappingFactor)
{
    return FloatPoint(floorToDevicePixel(point.x(), pixelSnappingFactor), floorToDevicePixel(point.y(), pixelSnappingFactor));
}

inline FloatPoint ceilPointToDevicePixels(const LayoutPoint& point, float pixelSnappingFactor)
{
    return FloatPoint(ceilToDevicePixel(point.x(), pixelSnappingFactor), ceilToDevicePixel(point.y(), pixelSnappingFactor));
}

inline FloatSize snapSizeToDevicePixel(const LayoutSize& size, const LayoutPoint& location, float pixelSnappingFactor)
{
    auto snap = [&] (LayoutUnit a, LayoutUnit b) {
        LayoutUnit fraction = b.fraction();
        return roundToDevicePixel(fraction + a, pixelSnappingFactor) - roundToDevicePixel(fraction, pixelSnappingFactor);
    };
    return FloatSize(snap(size.width(), location.x()), snap(size.height(), location.y()));
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const LayoutPoint&);

} // namespace WebCore

