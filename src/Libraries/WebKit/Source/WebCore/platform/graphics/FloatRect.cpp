/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#include "FloatRect.h"

#include "FloatConversion.h"
#include "IntRect.h"
#include <algorithm>
#include <math.h>
#include <wtf/MathExtras.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

FloatRect::FloatRect(const IntRect& r)
    : m_location(r.location())
    , m_size(r.size())
{
}

FloatRect FloatRect::narrowPrecision(double x, double y, double width, double height)
{
    return FloatRect(narrowPrecisionToFloat(x), narrowPrecisionToFloat(y), narrowPrecisionToFloat(width), narrowPrecisionToFloat(height));
}

bool FloatRect::isExpressibleAsIntRect() const
{
    return isWithinIntRange(x()) && isWithinIntRange(y())
        && isWithinIntRange(width()) && isWithinIntRange(height())
        && isWithinIntRange(maxX()) && isWithinIntRange(maxY());
}

bool FloatRect::inclusivelyIntersects(const FloatRect& other) const
{
    return width() >= 0 && height() >= 0 && other.width() >= 0 && other.height() >= 0
        && x() <= other.maxX() && other.x() <= maxX() && y() <= other.maxY() && other.y() <= maxY();
}

bool FloatRect::intersects(const FloatRect& other) const
{
    // Checking emptiness handles negative widths and heights as well as zero.
    return !isEmpty() && !other.isEmpty()
        && x() < other.maxX() && other.x() < maxX()
        && y() < other.maxY() && other.y() < maxY();
}

bool FloatRect::contains(const FloatRect& other) const
{
    return x() <= other.x() && maxX() >= other.maxX()
        && y() <= other.y() && maxY() >= other.maxY();
}

bool FloatRect::contains(const FloatPoint& point, ContainsMode containsMode) const
{
    if (containsMode == InsideOrOnStroke)
        return contains(point.x(), point.y());
    return x() < point.x() && maxX() > point.x() && y() < point.y() && maxY() > point.y();
}

void FloatRect::intersect(const FloatRect& other)
{
    float l = std::max(x(), other.x());
    float t = std::max(y(), other.y());
    float r = std::min(maxX(), other.maxX());
    float b = std::min(maxY(), other.maxY());

    // Return a clean empty rectangle for non-intersecting cases.
    if (l >= r || t >= b) {
        l = 0;
        t = 0;
        r = 0;
        b = 0;
    }

    setLocationAndSizeFromEdges(l, t, r, b);
}

bool FloatRect::edgeInclusiveIntersect(const FloatRect& other)
{
    FloatPoint newLocation(std::max(x(), other.x()), std::max(y(), other.y()));
    FloatPoint newMaxPoint(std::min(maxX(), other.maxX()), std::min(maxY(), other.maxY()));

    bool intersects = true;

    // Return a clean empty rectangle for non-intersecting cases.
    if (newLocation.x() > newMaxPoint.x() || newLocation.y() > newMaxPoint.y()) {
        newLocation = { };
        newMaxPoint = { };
        intersects = false;
    }

    m_location = newLocation;
    m_size = newMaxPoint - newLocation;
    return intersects;
}

void FloatRect::unite(const FloatRect& other)
{
    // Handle empty special cases first.
    if (other.isEmpty())
        return;
    if (isEmpty()) {
        *this = other;
        return;
    }

    uniteEvenIfEmpty(other);
}

void FloatRect::uniteEvenIfEmpty(const FloatRect& other)
{
    float minX = std::min(x(), other.x());
    float minY = std::min(y(), other.y());
    float maxX = std::max(this->maxX(), other.maxX());
    float maxY = std::max(this->maxY(), other.maxY());

    setLocationAndSizeFromEdges(minX, minY, maxX, maxY);
}

void FloatRect::uniteIfNonZero(const FloatRect& other)
{
    // Handle empty special cases first.
    if (other.isZero())
        return;
    if (isZero()) {
        *this = other;
        return;
    }

    uniteEvenIfEmpty(other);
}

void FloatRect::extend(const FloatPoint& p)
{
    float minX = std::min(x(), p.x());
    float minY = std::min(y(), p.y());
    float maxX = std::max(this->maxX(), p.x());
    float maxY = std::max(this->maxY(), p.y());

    setLocationAndSizeFromEdges(minX, minY, maxX, maxY);
}

void FloatRect::scale(float sx, float sy)
{
    m_location.setX(x() * sx);
    m_location.setY(y() * sy);
    m_size.setWidth(width() * sx);
    m_size.setHeight(height() * sy);
}

FloatRect normalizeRect(const FloatRect& rect)
{
    return FloatRect(std::min(rect.x(), rect.maxX()),
        std::min(rect.y(), rect.maxY()),
        std::max(rect.width(), -rect.width()),
        std::max(rect.height(), -rect.height()));
}

FloatRect encloseRectToDevicePixels(const FloatRect& rect, float deviceScaleFactor)
{
    FloatPoint location = floorPointToDevicePixels(rect.minXMinYCorner(), deviceScaleFactor);
    FloatPoint maxPoint = ceilPointToDevicePixels(rect.maxXMaxYCorner(), deviceScaleFactor);
    return FloatRect(location, maxPoint - location);
}

IntRect enclosingIntRect(const FloatRect& rect)
{
    FloatPoint location = flooredIntPoint(rect.minXMinYCorner());
    FloatPoint maxPoint = ceiledIntPoint(rect.maxXMaxYCorner());
    return IntRect(IntPoint(location), IntSize(maxPoint - location));
}

IntRect enclosingIntRectPreservingEmptyRects(const FloatRect& rect)
{
    // Empty rects with fractional x, y values turn into non-empty rects when converting to enclosing.
    // We want to ensure that empty rects stay empty after the conversion, since some callers
    // prefer this behavior.
    FloatPoint location = flooredIntPoint(rect.minXMinYCorner());
    if (rect.isEmpty())
        return IntRect(IntPoint(location), { });
    FloatPoint maxPoint = ceiledIntPoint(rect.maxXMaxYCorner());
    return IntRect(IntPoint(location), IntSize(maxPoint - location));
}

IntRect roundedIntRect(const FloatRect& rect)
{
    return IntRect(roundedIntPoint(rect.location()), roundedIntSize(rect.size()));
}

TextStream& operator<<(TextStream& ts, const FloatRect &r)
{
    if (ts.hasFormattingFlag(TextStream::Formatting::SVGStyleRect)) {
        // FIXME: callers should use the NumberRespectingIntegers flag.
        return ts << "at (" << TextStream::FormatNumberRespectingIntegers(r.x()) << "," << TextStream::FormatNumberRespectingIntegers(r.y())
            << ") size " << TextStream::FormatNumberRespectingIntegers(r.width()) << "x" << TextStream::FormatNumberRespectingIntegers(r.height());
    }

    return ts << r.location() << " " << r.size();
}

Ref<JSON::Object> FloatRect::toJSONObject() const
{
    auto object = JSON::Object::create();

    object->setObject("location"_s, m_location.toJSONObject());
    object->setObject("size"_s, m_size.toJSONObject());

    return object;
}

String FloatRect::toJSONString() const
{
    return toJSONObject()->toJSONString();
}

}
