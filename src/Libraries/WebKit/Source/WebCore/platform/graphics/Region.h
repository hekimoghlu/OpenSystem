/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#ifndef Region_h
#define Region_h

#include "IntRect.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/PointerComparison.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(Region);
class Region {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Region);
public:
    WEBCORE_EXPORT Region();
    WEBCORE_EXPORT Region(const IntRect&);

    WEBCORE_EXPORT Region(const Region&);
    WEBCORE_EXPORT Region(Region&&);

    WEBCORE_EXPORT ~Region();

    WEBCORE_EXPORT Region& operator=(const Region&);
    WEBCORE_EXPORT Region& operator=(Region&&);

    IntRect bounds() const { return m_bounds; }
    bool isEmpty() const { return m_bounds.isEmpty(); }
    bool isRect() const { return !m_shape; }

    WEBCORE_EXPORT Vector<IntRect, 1> rects() const;

    WEBCORE_EXPORT void unite(const Region&);
    WEBCORE_EXPORT void intersect(const Region&);
    WEBCORE_EXPORT void subtract(const Region&);

    WEBCORE_EXPORT void translate(const IntSize&);

    // Returns true if the query region is a subset of this region.
    WEBCORE_EXPORT bool contains(const Region&) const;

    WEBCORE_EXPORT bool contains(const IntPoint&) const;

    // Returns true if the query region intersects any part of this region.
    WEBCORE_EXPORT bool intersects(const Region&) const;

    WEBCORE_EXPORT uint64_t totalArea() const;

    unsigned gridSize() const { return m_shape ? m_shape->gridSize() : 0; }

    struct Span {
        int y { 0 };
        size_t segmentIndex { 0 };

        friend bool operator==(const Span&, const Span&) = default;
    };

    class Shape {
        WTF_MAKE_TZONE_ALLOCATED_EXPORT(Shape, WEBCORE_EXPORT);
    public:
        Shape() = default;
        WEBCORE_EXPORT Shape(const IntRect&);

        IntRect bounds() const;
        bool isEmpty() const { return m_spans.isEmpty(); }
        bool isRect() const { return m_spans.size() <= 2 && m_segments.size() <= 2; }
        unsigned gridSize() const { return m_spans.size() * m_segments.size(); }

        std::span<const Span> spans() const { return m_spans.span(); }
        std::span<const int> segments(std::span<const Span>) const;

        static Shape unionShapes(const Shape& shape1, const Shape& shape2);
        static Shape intersectShapes(const Shape& shape1, const Shape& shape2);
        static Shape subtractShapes(const Shape& shape1, const Shape& shape2);

        WEBCORE_EXPORT void translate(const IntSize&);

        struct CompareContainsOperation;
        struct CompareIntersectsOperation;

        template<typename CompareOperation>
        static bool compareShapes(const Shape& shape1, const Shape& shape2);

        WEBCORE_EXPORT static bool isValidShape(std::span<const int> segments, std::span<const Span> spans);

        static Shape createForTesting(Vector<int, 32>&& segments, Vector<Span, 16>&& spans) { return Shape { WTFMove(segments), WTFMove(spans) }; }
        std::pair<Vector<int, 32>, Vector<Span, 16>> dataForTesting() const { return { m_segments, m_spans }; }
    private:
        WEBCORE_EXPORT Shape(Vector<int, 32>&&, Vector<Span, 16>&&);
        struct UnionOperation;
        struct IntersectOperation;
        struct SubtractOperation;

        template<typename Operation>
        static Shape shapeOperation(const Shape& shape1, const Shape& shape2);

        void appendSpan(int y);
        void appendSpan(int y, std::span<const int> segments);
        void appendSpans(const Shape&, std::span<const Span> spans);

        bool canCoalesce(std::span<const int> segments);

        Vector<int, 32> m_segments;
        Vector<Span, 16> m_spans;
        friend struct IPC::ArgumentCoder<WebCore::Region::Shape, void>;
        friend bool operator==(const Shape&, const Shape&) = default;
        WEBCORE_EXPORT friend WTF::TextStream& operator<<(WTF::TextStream&, const Shape&);
    };
    static Region createForTesting(Shape&& shape) { return Region { WTFMove(shape) }; }
    Shape dataForTesting() const { return data(); }
private:
    friend struct IPC::ArgumentCoder<WebCore::Region, void>;
    explicit Region(Shape&& shape) { setShape(WTFMove(shape)); }
    Shape data() const;

    std::unique_ptr<Shape> copyShape() const { return m_shape ? makeUnique<Shape>(*m_shape) : nullptr; }
    WEBCORE_EXPORT void setShape(Shape&&);

    IntRect m_bounds;
    std::unique_ptr<Shape> m_shape;

    friend bool operator==(const Region&, const Region&);
};

inline Region::Shape Region::data() const
{
    if (m_shape)
        return *m_shape;
    return m_bounds;
}

static inline Region intersect(const Region& a, const Region& b)
{
    Region result(a);
    result.intersect(b);

    return result;
}

static inline Region subtract(const Region& a, const Region& b)
{
    Region result(a);
    result.subtract(b);

    return result;
}

static inline Region translate(const Region& region, const IntSize& offset)
{
    Region result(region);
    result.translate(offset);

    return result;
}

inline bool operator==(const Region& a, const Region& b)
{
    return a.m_bounds == b.m_bounds && arePointingToEqualData(a.m_shape, b.m_shape);
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const Region&);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const Region::Shape&);

} // namespace WebCore

#endif // Region_h
