/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include "IntRect.h"

#include "FloatRect.h"
#include "LayoutRect.h"
#include <algorithm>
#include <wtf/CheckedArithmetic.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IntRect);

IntRect::IntRect(const FloatRect& r)
    : m_location(clampToInteger(r.x()), clampToInteger(r.y()))
    , m_size(clampToInteger(r.width()), clampToInteger(r.height()))
{
}

IntRect::IntRect(const LayoutRect& r)
    : m_location(r.x(), r.y())
    , m_size(r.width(), r.height())
{
}

bool IntRect::intersects(const IntRect& other) const
{
    // Checking emptiness handles negative widths as well as zero.
    return !isEmpty() && !other.isEmpty()
        && x() < other.maxX() && other.x() < maxX()
        && y() < other.maxY() && other.y() < maxY();
}

bool IntRect::contains(const IntRect& other) const
{
    return x() <= other.x() && maxX() >= other.maxX()
        && y() <= other.y() && maxY() >= other.maxY();
}

void IntRect::intersect(const IntRect& other)
{
    int l = std::max(x(), other.x());
    int t = std::max(y(), other.y());
    int r = std::min(maxX(), other.maxX());
    int b = std::min(maxY(), other.maxY());

    // Return a clean empty rectangle for non-intersecting cases.
    if (l >= r || t >= b) {
        l = 0;
        t = 0;
        r = 0;
        b = 0;
    }

    m_location.setX(l);
    m_location.setY(t);
    m_size.setWidth(r - l);
    m_size.setHeight(b - t);
}

void IntRect::unite(const IntRect& other)
{
    // Handle empty special cases first.
    if (other.isEmpty())
        return;
    if (isEmpty()) {
        *this = other;
        return;
    }

    int l = std::min(x(), other.x());
    int t = std::min(y(), other.y());
    int r = std::max(maxX(), other.maxX());
    int b = std::max(maxY(), other.maxY());

    m_location.setX(l);
    m_location.setY(t);
    m_size.setWidth(r - l);
    m_size.setHeight(b - t);
}

void IntRect::uniteIfNonZero(const IntRect& other)
{
    // Handle empty special cases first.
    if (other.isZero())
        return;
    if (isZero()) {
        *this = other;
        return;
    }

    int left = std::min(x(), other.x());
    int top = std::min(y(), other.y());
    int right = std::max(maxX(), other.maxX());
    int bottom = std::max(maxY(), other.maxY());

    m_location.setX(left);
    m_location.setY(top);
    m_size.setWidth(right - left);
    m_size.setHeight(bottom - top);
}

void IntRect::scale(float s)
{
    m_location.setX((int)(x() * s));
    m_location.setY((int)(y() * s));
    m_size.setWidth((int)(width() * s));
    m_size.setHeight((int)(height() * s));
}

static inline int distanceToInterval(int pos, int start, int end)
{
    if (pos < start)
        return start - pos;
    if (pos > end)
        return end - pos;
    return 0;
}

IntSize IntRect::differenceToPoint(const IntPoint& point) const
{
    int xdistance = distanceToInterval(point.x(), x(), maxX());
    int ydistance = distanceToInterval(point.y(), y(), maxY());
    return IntSize(xdistance, ydistance);
}

bool IntRect::isValid() const
{
    CheckedInt32 max = m_location.x();
    max += m_size.width();
    if (max.hasOverflowed())
        return false;
    max = m_location.y();
    max += m_size.height();
    return !max.hasOverflowed();
}

IntRect IntRect::toRectWithExtentsClippedToNumericLimits() const
{
    using T = int32_t;
    IntRect clippedRect { *this };
    constexpr auto max = std::numeric_limits<T>::max();
    if (sumOverflows<T>(x(), width()))
        clippedRect.setWidth(max - x());
    if (sumOverflows<T>(y(), height()))
        clippedRect.setHeight(max - y());
    return clippedRect;
}

TextStream& operator<<(TextStream& ts, const IntRect& r)
{
    if (ts.hasFormattingFlag(TextStream::Formatting::SVGStyleRect))
        return ts << "at (" << r.x() << "," << r.y() << ") size " << r.width() << "x" << r.height();

    return ts << r.location() << " " << r.size();
}

} // namespace WebCore
