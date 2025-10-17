/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include "LayoutShape.h"
#include "ShapeInterval.h"
#include <wtf/Assertions.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class RasterShapeIntervals {
    WTF_MAKE_TZONE_ALLOCATED(RasterShapeIntervals);
public:
    explicit RasterShapeIntervals(unsigned size, int offset = 0)
        : m_intervals(clampTo<int>(size))
        , m_offset(offset)
    {
    }

    void initializeBounds();
    const IntRect& bounds() const { return m_bounds; }
    bool isEmpty() const { return m_bounds.isEmpty(); }

    IntShapeInterval& intervalAt(int y)
    {
        ASSERT(y + m_offset >= 0 && static_cast<unsigned>(y + m_offset) < m_intervals.size());
        return m_intervals[y + m_offset];
    }

    const IntShapeInterval& intervalAt(int y) const
    {
        ASSERT(y + m_offset >= 0 && static_cast<unsigned>(y + m_offset) < m_intervals.size());
        return m_intervals[y + m_offset];
    }

    std::unique_ptr<RasterShapeIntervals> computeShapeMarginIntervals(int shapeMargin) const;
    void buildBoundsPath(Path&) const;

private:
    int size() const { return m_intervals.size(); }
    int offset() const { return m_offset; }
    int minY() const { return -m_offset; }
    int maxY() const { return -m_offset + m_intervals.size(); }

    IntRect m_bounds;
    Vector<IntShapeInterval> m_intervals;
    int m_offset;
};

class RasterLayoutShape final : public LayoutShape {
    WTF_MAKE_NONCOPYABLE(RasterLayoutShape);
public:
    RasterLayoutShape(std::unique_ptr<RasterShapeIntervals> intervals, const IntSize& marginRectSize)
        : m_intervals(WTFMove(intervals))
        , m_marginRectSize(marginRectSize)
    {
        m_intervals->initializeBounds();
    }

    LayoutRect shapeMarginLogicalBoundingBox() const override { return static_cast<LayoutRect>(marginIntervals().bounds()); }
    bool isEmpty() const override { return m_intervals->isEmpty(); }
    LineSegment getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const override;

    void buildDisplayPaths(DisplayPaths& paths) const override
    {
        m_intervals->buildBoundsPath(paths.shape);
        if (shapeMargin())
            marginIntervals().buildBoundsPath(paths.marginShape);
    }

private:
    const RasterShapeIntervals& marginIntervals() const;

    std::unique_ptr<RasterShapeIntervals> m_intervals;
    mutable std::unique_ptr<RasterShapeIntervals> m_marginIntervals;
    IntSize m_marginRectSize;
};

} // namespace WebCore
