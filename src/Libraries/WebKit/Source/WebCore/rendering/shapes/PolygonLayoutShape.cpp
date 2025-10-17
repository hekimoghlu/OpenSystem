/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#include "PolygonLayoutShape.h"

#include <wtf/MathExtras.h>

namespace WebCore {

static inline FloatSize inwardEdgeNormal(const FloatPolygonEdge& edge)
{
    FloatSize edgeDelta = edge.vertex2() - edge.vertex1();
    if (!edgeDelta.width())
        return FloatSize((edgeDelta.height() > 0 ? -1 : 1), 0);
    if (!edgeDelta.height())
        return FloatSize(0, (edgeDelta.width() > 0 ? 1 : -1));
    float edgeLength = edgeDelta.diagonalLength();
    return FloatSize(-edgeDelta.height() / edgeLength, edgeDelta.width() / edgeLength);
}

static inline FloatSize outwardEdgeNormal(const FloatPolygonEdge& edge)
{
    return -inwardEdgeNormal(edge);
}

float OffsetPolygonEdge::xIntercept(float y) const
{
    ASSERT(y >= minY() && y <= maxY());

    if (vertex1().y() == vertex2().y() || vertex1().x() == vertex2().x())
        return minX();
    if (y == minY())
        return vertex1().y() < vertex2().y() ? vertex1().x() : vertex2().x();
    if (y == maxY())
        return vertex1().y() > vertex2().y() ? vertex1().x() : vertex2().x();

    return vertex1().x() + ((y - vertex1().y()) * (vertex2().x() - vertex1().x()) / (vertex2().y() - vertex1().y()));
}

FloatShapeInterval OffsetPolygonEdge::clippedEdgeXRange(float y1, float y2) const
{
    if (!overlapsYRange(y1, y2) || (y1 == maxY() && minY() <= y1) || (y2 == minY() && maxY() >= y2))
        return FloatShapeInterval();

    if (isWithinYRange(y1, y2))
        return FloatShapeInterval(minX(), maxX());

    // Clip the edge line segment to the vertical range y1,y2 and then return
    // the clipped line segment's horizontal range.

    FloatPoint minYVertex;
    FloatPoint maxYVertex;
    if (vertex1().y() < vertex2().y()) {
        minYVertex = vertex1();
        maxYVertex = vertex2();
    } else {
        minYVertex = vertex2();
        maxYVertex = vertex1();
    }
    float xForY1 = (minYVertex.y() < y1) ? xIntercept(y1) : minYVertex.x();
    float xForY2 = (maxYVertex.y() > y2) ? xIntercept(y2) : maxYVertex.x();
    return FloatShapeInterval(std::min(xForY1, xForY2), std::max(xForY1, xForY2));
}

static float circleXIntercept(float y, float radius)
{
    ASSERT(radius > 0);
    return radius * sqrt(1 - (y * y) / (radius * radius));
}

static FloatShapeInterval clippedCircleXRange(const FloatPoint& center, float radius, float y1, float y2)
{
    if (y1 >= center.y() + radius || y2 <= center.y() - radius)
        return FloatShapeInterval();

    if (center.y() >= y1 && center.y() <= y2)
        return FloatShapeInterval(center.x() - radius, center.x() + radius);

    // Clip the circle to the vertical range y1,y2 and return the extent of the clipped circle's
    // projection on the X axis

    float xi =  circleXIntercept((y2 < center.y() ? y2 : y1) - center.y(), radius);
    return FloatShapeInterval(center.x() - xi, center.x() + xi);
}

LayoutRect PolygonLayoutShape::shapeMarginLogicalBoundingBox() const
{
    FloatRect box = m_polygon.boundingBox();
    box.inflate(shapeMargin());
    return LayoutRect(box);
}

LineSegment PolygonLayoutShape::getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const
{
    float y1 = logicalTop;
    float y2 = logicalTop + logicalHeight;

    if (m_polygon.isEmpty() || !m_polygon.boundingBox().overlapsYRange(y1 - shapeMargin(), y2 + shapeMargin()))
        return { };

    FloatShapeInterval excludedInterval;
    for (const FloatPolygonEdge& edge : m_polygon.overlappingEdges(y1 - shapeMargin(), y2 + shapeMargin())) {
        if (edge.maxY() == edge.minY())
            continue;
        if (!shapeMargin())
            excludedInterval.unite(OffsetPolygonEdge(edge, FloatSize()).clippedEdgeXRange(y1, y2));
        else {
            excludedInterval.unite(OffsetPolygonEdge(edge, outwardEdgeNormal(edge) * shapeMargin()).clippedEdgeXRange(y1, y2));
            excludedInterval.unite(OffsetPolygonEdge(edge, inwardEdgeNormal(edge) * shapeMargin()).clippedEdgeXRange(y1, y2));
            excludedInterval.unite(clippedCircleXRange(edge.vertex1(), shapeMargin(), y1, y2));
            excludedInterval.unite(clippedCircleXRange(edge.vertex2(), shapeMargin(), y1, y2));
        }
    }

    if (excludedInterval.isEmpty())
        return { };

    if (writingMode().isBidiRTL())
        return { std::max(0.f, m_boxLogicalWidth - excludedInterval.x2()), std::max(0.f, m_boxLogicalWidth - excludedInterval.x1()) };

    return { excludedInterval.x1(), excludedInterval.x2() };
}

void PolygonLayoutShape::buildDisplayPaths(DisplayPaths& paths) const
{
    if (m_polygon.isEmpty())
        return;

    paths.shape.moveTo(m_polygon.vertexAt(0));
    for (unsigned i = 1; i < m_polygon.numberOfVertices(); i++)
        paths.shape.addLineTo(m_polygon.vertexAt(i));

    paths.shape.closeSubpath();
}

} // namespace WebCore
