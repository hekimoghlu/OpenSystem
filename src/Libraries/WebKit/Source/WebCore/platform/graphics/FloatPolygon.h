/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#include "FloatRect.h"
#include "PODIntervalTree.h"
#include "WindRule.h"
#include <memory>
#include <wtf/Vector.h>

namespace WebCore {

class FloatPolygonEdge;

typedef PODIntervalTree<float, FloatPolygonEdge*> EdgeIntervalTree;

class FloatPolygon {
public:
    FloatPolygon(Vector<FloatPoint>&& vertices);

    const FloatPoint& vertexAt(unsigned index) const { return m_vertices[index]; }
    unsigned numberOfVertices() const { return m_vertices.size(); }

    const FloatPolygonEdge& edgeAt(unsigned index) const { return m_edges[index]; }
    unsigned numberOfEdges() const { return m_edges.size(); }

    FloatRect boundingBox() const { return m_boundingBox; }
    Vector<std::reference_wrapper<const FloatPolygonEdge>> overlappingEdges(float minY, float maxY) const;
    bool contains(const FloatPoint&) const;
    bool isEmpty() const { return m_empty; }

private:
    Vector<FloatPoint> m_vertices;
    FloatRect m_boundingBox;
    bool m_empty;
    Vector<FloatPolygonEdge> m_edges;
    EdgeIntervalTree m_edgeTree; // Each EdgeIntervalTree node stores minY, maxY, and a ("UserData") pointer to a FloatPolygonEdge.

};

class VertexPair {
public:
    virtual ~VertexPair() = default;

    virtual const FloatPoint& vertex1() const = 0;
    virtual const FloatPoint& vertex2() const = 0;

    float minX() const { return std::min(vertex1().x(), vertex2().x()); }
    float minY() const { return std::min(vertex1().y(), vertex2().y()); }
    float maxX() const { return std::max(vertex1().x(), vertex2().x()); }
    float maxY() const { return std::max(vertex1().y(), vertex2().y()); }

    bool intersection(const VertexPair&, FloatPoint&) const;
};

class FloatPolygonEdge : public VertexPair {
    friend class FloatPolygon;
public:
    const FloatPoint& vertex1() const override
    {
        ASSERT(m_polygon);
        return m_polygon->vertexAt(m_vertexIndex1);
    }

    const FloatPoint& vertex2() const override
    {
        ASSERT(m_polygon);
        return m_polygon->vertexAt(m_vertexIndex2);
    }

    const FloatPolygonEdge& previousEdge() const
    {
        ASSERT(m_polygon && m_polygon->numberOfEdges() > 1);
        return m_polygon->edgeAt((m_edgeIndex + m_polygon->numberOfEdges() - 1) % m_polygon->numberOfEdges());
    }

    const FloatPolygonEdge& nextEdge() const
    {
        ASSERT(m_polygon && m_polygon->numberOfEdges() > 1);
        return m_polygon->edgeAt((m_edgeIndex + 1) % m_polygon->numberOfEdges());
    }

    const FloatPolygon* polygon() const { return m_polygon; }
    unsigned vertexIndex1() const { return m_vertexIndex1; }
    unsigned vertexIndex2() const { return m_vertexIndex2; }
    unsigned edgeIndex() const { return m_edgeIndex; }

private:
    // Edge vertex index1 is less than index2, except the last edge, where index2 is 0. When a polygon edge
    // is defined by 3 or more colinear vertices, index2 can be the index of the last colinear vertex.
    unsigned m_vertexIndex1;
    unsigned m_vertexIndex2;
    unsigned m_edgeIndex;
    const FloatPolygon* m_polygon;
};

#ifndef NDEBUG
TextStream& operator<<(TextStream&, const FloatPolygonEdge&);
#endif

} // namespace WebCore
