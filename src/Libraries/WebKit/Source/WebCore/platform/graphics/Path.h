/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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

#include "PathElement.h"
#include "PathImpl.h"
#include "PathSegment.h"
#include "PlatformPath.h"
#include "WindRule.h"
#include <wtf/DataRef.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GraphicsContext;
class PathTraversalState;
class RoundedRect;

class Path {
    WTF_MAKE_TZONE_ALLOCATED(Path);
public:
    Path() = default;
    WEBCORE_EXPORT Path(PathSegment&&);
    WEBCORE_EXPORT Path(Vector<PathSegment>&&);
    explicit Path(const Vector<FloatPoint>& points);
    Path(Ref<PathImpl>&&);

    WEBCORE_EXPORT Path(const Path&);
    Path(Path&&) = default;
    Path& operator=(const Path&) = default;
    Path& operator=(Path&&) = default;

    WEBCORE_EXPORT bool definitelyEqual(const Path&) const;

    WEBCORE_EXPORT void moveTo(const FloatPoint&);

    WEBCORE_EXPORT void addLineTo(const FloatPoint&);
    WEBCORE_EXPORT void addQuadCurveTo(const FloatPoint& controlPoint, const FloatPoint& endPoint);
    WEBCORE_EXPORT void addBezierCurveTo(const FloatPoint& controlPoint1, const FloatPoint& controlPoint2, const FloatPoint& endPoint);
    void addArcTo(const FloatPoint& point1, const FloatPoint& point2, float radius);

    void addArc(const FloatPoint&, float radius, float startAngle, float endAngle, RotationDirection);
    void addEllipse(const FloatPoint&, float radiusX, float radiusY, float rotation, float startAngle, float endAngle, RotationDirection);
    void addEllipseInRect(const FloatRect&);
    WEBCORE_EXPORT void addRect(const FloatRect&);
    WEBCORE_EXPORT void addRoundedRect(const FloatRoundedRect&, PathRoundedRect::Strategy = PathRoundedRect::Strategy::PreferNative);
    void addRoundedRect(const FloatRect&, const FloatSize& roundingRadii, PathRoundedRect::Strategy = PathRoundedRect::Strategy::PreferNative);
    void addRoundedRect(const RoundedRect&);

    WEBCORE_EXPORT void closeSubpath();

    void addPath(const Path&, const AffineTransform&);

    void applySegments(const PathSegmentApplier&) const;
    WEBCORE_EXPORT void applyElements(const PathElementApplier&) const;
    void clear();

    void translate(const FloatSize& delta);
    void transform(const AffineTransform&);

    static constexpr float circleControlPoint() { return PathImpl::circleControlPoint(); }

    WEBCORE_EXPORT std::optional<PathSegment> singleSegment() const;
    std::optional<PathDataLine> singleDataLine() const;
    std::optional<PathRect> singleRect() const;
    std::optional<PathRoundedRect> singleRoundedRect() const;
    std::optional<PathArc> singleArc() const;
    std::optional<PathClosedArc> singleClosedArc() const;
    std::optional<PathDataQuadCurve> singleQuadCurve() const;
    std::optional<PathDataBezierCurve> singleBezierCurve() const;

    WEBCORE_EXPORT bool isEmpty() const;
    bool definitelySingleLine() const;
    WEBCORE_EXPORT PlatformPathPtr platformPath() const;

    const PathSegment* singleSegmentIfExists() const { return asSingle(); }
    WEBCORE_EXPORT const Vector<PathSegment>* segmentsIfExists() const;
    WEBCORE_EXPORT Vector<PathSegment> segments() const;

    float length() const;
    bool isClosed() const;
    bool hasSubpaths() const;
    FloatPoint currentPoint() const;
    PathTraversalState traversalStateAtLength(float length) const;
    FloatPoint pointAtLength(float length) const;

    bool contains(const FloatPoint&, WindRule = WindRule::NonZero) const;
    bool strokeContains(const FloatPoint&, const Function<void(GraphicsContext&)>& strokeStyleApplier) const;

    WEBCORE_EXPORT FloatRect fastBoundingRect() const;
    WEBCORE_EXPORT FloatRect boundingRect() const;
    FloatRect strokeBoundingRect(const Function<void(GraphicsContext&)>& strokeStyleApplier = { }) const;

    WEBCORE_EXPORT void ensureImplForTesting();

private:
    PlatformPathImpl& ensurePlatformPathImpl();
    PathImpl& setImpl(Ref<PathImpl>&&);
    PathImpl& ensureImpl();

    PathSegment* asSingle() { return std::get_if<PathSegment>(&m_data); }
    const PathSegment* asSingle() const { return std::get_if<PathSegment>(&m_data); }

    PathImpl* asImpl();
    const PathImpl* asImpl() const;

    const PathMoveTo* asSingleMoveTo() const;
    const PathArc* asSingleArc() const;

    std::variant<std::monostate, PathSegment, DataRef<PathImpl>> m_data;
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const Path&);

} // namespace WebCore
