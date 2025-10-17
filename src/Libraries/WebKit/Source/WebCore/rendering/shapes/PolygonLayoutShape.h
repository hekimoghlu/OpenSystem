/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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

#include "FloatPolygon.h"
#include "LayoutShape.h"
#include "ShapeInterval.h"

namespace WebCore {

class OffsetPolygonEdge final : public VertexPair {
public:
    OffsetPolygonEdge(const FloatPolygonEdge& edge, const FloatSize& offset)
        : m_vertex1(edge.vertex1() + offset)
        , m_vertex2(edge.vertex2() + offset)
    {
    }

    const FloatPoint& vertex1() const override { return m_vertex1; }
    const FloatPoint& vertex2() const override { return m_vertex2; }

    bool isWithinYRange(float y1, float y2) const { return y1 <= minY() && y2 >= maxY(); }
    bool overlapsYRange(float y1, float y2) const { return y2 >= minY() && y1 <= maxY(); }
    float xIntercept(float y) const;
    FloatShapeInterval clippedEdgeXRange(float y1, float y2) const;

private:
    FloatPoint m_vertex1;
    FloatPoint m_vertex2;
};

class PolygonLayoutShape : public LayoutShape {
    WTF_MAKE_NONCOPYABLE(PolygonLayoutShape);
public:
    PolygonLayoutShape(Vector<FloatPoint>&& vertices, float boxLogicalWidth)
        : m_polygon(WTFMove(vertices))
        , m_boxLogicalWidth(boxLogicalWidth)
    {
    }

    LayoutRect shapeMarginLogicalBoundingBox() const override;
    bool isEmpty() const override { return m_polygon.isEmpty(); }
    LineSegment getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const override;

    void buildDisplayPaths(DisplayPaths&) const override;

private:
    FloatPolygon m_polygon;
    float m_boxLogicalWidth { 0.f };
};

} // namespace WebCore
