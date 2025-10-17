/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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

#include "PathImpl.h"
#include "PathSegment.h"
#include <wtf/DataRef.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class PathStream final : public PathImpl {
    WTF_MAKE_TZONE_ALLOCATED(PathStream);
public:
    static Ref<PathStream> create();
    static Ref<PathStream> create(PathSegment&&);
    static Ref<PathStream> create(const Vector<FloatPoint>&);
    static Ref<PathStream> create(Vector<PathSegment>&&);

    bool definitelyEqual(const PathImpl&) const final;
    Ref<PathImpl> copy() const final;

    void add(PathMoveTo) final;
    void add(PathLineTo) final;
    void add(PathQuadCurveTo) final;
    void add(PathBezierCurveTo) final;
    void add(PathArcTo) final;
    void add(PathArc) final;
    void add(PathClosedArc) final;
    void add(PathEllipse) final;
    void add(PathEllipseInRect) final;
    void add(PathRect) final;
    void add(PathRoundedRect) final;
    void add(PathCloseSubpath) final;

    const Vector<PathSegment>& segments() const { return m_segments; }

    void applySegments(const PathSegmentApplier&) const final;
    bool applyElements(const PathElementApplier&) const final;

    bool transform(const AffineTransform&) final;

    FloatRect fastBoundingRect() const final;
    FloatRect boundingRect() const final;

    bool hasSubpaths() const final;

    static FloatRect computeFastBoundingRect(std::span<const PathSegment>);
    static FloatRect computeBoundingRect(std::span<const PathSegment>);
    static bool computeHasSubpaths(std::span<const PathSegment>);

private:
    PathStream() = default;
    PathStream(PathSegment&&);
    PathStream(Vector<PathSegment>&&);
    PathStream(const Vector<PathSegment>&);

    static Ref<PathStream> create(const Vector<PathSegment>&);

    const PathMoveTo* lastIfMoveTo() const;

    bool isPathStream() const final { return true; }

    template<typename DataType>
    std::optional<DataType> singleDataType() const;

    std::optional<PathSegment> singleSegment() const final;
    std::optional<PathDataLine> singleDataLine() const final;
    std::optional<PathRect> singleRect() const final;
    std::optional<PathRoundedRect> singleRoundedRect() const final;
    std::optional<PathArc> singleArc() const final;
    std::optional<PathClosedArc> singleClosedArc() const final;
    std::optional<PathDataQuadCurve> singleQuadCurve() const final;
    std::optional<PathDataBezierCurve> singleBezierCurve() const final;

    bool isEmpty() const final { return m_segments.isEmpty(); }

    bool isClosed() const final;
    FloatPoint currentPoint() const final;

    Vector<PathSegment>& segments() { return m_segments; }

    Vector<PathSegment> m_segments;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PathStream)
    static bool isType(const WebCore::PathImpl& pathImpl) { return pathImpl.isPathStream(); }
SPECIALIZE_TYPE_TRAITS_END()
