/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#include "RasterLayoutShape.h"

#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RasterShapeIntervals);

class MarginIntervalGenerator {
public:
    explicit MarginIntervalGenerator(unsigned radius);
    void set(int y, const IntShapeInterval&);
    IntShapeInterval intervalAt(int y) const;

private:
    Vector<int> m_xIntercepts;
    int m_y;
    int m_x1;
    int m_x2;
};

MarginIntervalGenerator::MarginIntervalGenerator(unsigned radius)
    : m_xIntercepts(radius + 1)
    , m_y(0)
    , m_x1(0)
    , m_x2(0)
{
    unsigned radiusSquared = radius * radius;
    for (unsigned y = 0; y <= radius; y++)
        m_xIntercepts[y] = sqrt(static_cast<double>(radiusSquared - y * y));
}

void MarginIntervalGenerator::set(int y, const IntShapeInterval& interval)
{
    ASSERT(y >= 0 && interval.x1() >= 0);
    m_y = y;
    m_x1 = interval.x1();
    m_x2 = interval.x2();
}

IntShapeInterval MarginIntervalGenerator::intervalAt(int y) const
{
    unsigned xInterceptsIndex = std::abs(y - m_y);
    int dx = (xInterceptsIndex >= m_xIntercepts.size()) ? 0 : m_xIntercepts[xInterceptsIndex];
    return IntShapeInterval(m_x1 - dx, m_x2 + dx);
}

std::unique_ptr<RasterShapeIntervals> RasterShapeIntervals::computeShapeMarginIntervals(int shapeMargin) const
{
    int marginIntervalsSize = (offset() > shapeMargin) ? size() : size() - offset() * 2 + shapeMargin * 2;
    auto result = makeUnique<RasterShapeIntervals>(marginIntervalsSize, std::max(shapeMargin, offset()));
    MarginIntervalGenerator marginIntervalGenerator(shapeMargin);

    for (int y = bounds().y(); y < bounds().maxY(); ++y) {
        const IntShapeInterval& intervalAtY = intervalAt(y);
        if (intervalAtY.isEmpty())
            continue;

        marginIntervalGenerator.set(y, intervalAtY);
        int marginY0 = std::max(minY(), y - shapeMargin);
        int marginY1 = std::min(maxY(), y + shapeMargin + 1);

        for (int marginY = y - 1; marginY >= marginY0; --marginY) {
            if (marginY > bounds().y() && intervalAt(marginY).contains(intervalAtY))
                break;
            result->intervalAt(marginY).unite(marginIntervalGenerator.intervalAt(marginY));
        }

        result->intervalAt(y).unite(marginIntervalGenerator.intervalAt(y));

        for (int marginY = y + 1; marginY < marginY1; ++marginY) {
            if (marginY < bounds().maxY() && intervalAt(marginY).contains(intervalAtY))
                break;
            result->intervalAt(marginY).unite(marginIntervalGenerator.intervalAt(marginY));
        }
    }

    result->initializeBounds();
    return result;
}

void RasterShapeIntervals::initializeBounds()
{
    m_bounds = IntRect();
    for (int y = minY(); y < maxY(); ++y) {
        const IntShapeInterval& intervalAtY = intervalAt(y);
        if (intervalAtY.isEmpty())
            continue;
        m_bounds.unite(IntRect(intervalAtY.x1(), y, intervalAtY.width(), 1));
    }
}

void RasterShapeIntervals::buildBoundsPath(Path& path) const
{
    for (int y = bounds().y(); y < bounds().maxY(); y++) {
        if (intervalAt(y).isEmpty())
            continue;

        IntShapeInterval extent = intervalAt(y);
        int endY = y + 1;
        for (; endY < bounds().maxY(); endY++) {
            if (intervalAt(endY).isEmpty() || intervalAt(endY) != extent)
                break;
        }
        path.addRect(FloatRect(extent.x1(), y, extent.width(), endY - y));
        y = endY - 1;
    }
}

const RasterShapeIntervals& RasterLayoutShape::marginIntervals() const
{
    ASSERT(shapeMargin() >= 0);
    if (!shapeMargin())
        return *m_intervals;

    int shapeMarginInt = clampToPositiveInteger(ceil(shapeMargin()));
    int maxShapeMarginInt = std::max(m_marginRectSize.width(), m_marginRectSize.height()) * sqrt(2);
    if (!m_marginIntervals)
        m_marginIntervals = m_intervals->computeShapeMarginIntervals(std::min(shapeMarginInt, maxShapeMarginInt));

    return *m_marginIntervals;
}

LineSegment RasterLayoutShape::getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const
{
    const RasterShapeIntervals& intervals = marginIntervals();
    if (intervals.isEmpty())
        return LineSegment();

    int y1 = logicalTop;
    int y2 = logicalTop + logicalHeight;
    ASSERT(y2 >= y1);
    if (y2 < intervals.bounds().y() || y1 >= intervals.bounds().maxY())
        return LineSegment();

    y1 = std::max(y1, intervals.bounds().y());
    y2 = std::min(y2, intervals.bounds().maxY());
    IntShapeInterval excludedInterval;

    if (y1 == y2)
        excludedInterval = intervals.intervalAt(y1);
    else {
        for (int y = y1; y < y2;  y++)
            excludedInterval.unite(intervals.intervalAt(y));
    }

    return LineSegment(excludedInterval.x1(), excludedInterval.x2());
}

} // namespace WebCore
