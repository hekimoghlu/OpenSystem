/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

#if USE(SKIA)

#include "PathImpl.h"
#include "PlatformPath.h"
#include "WindRule.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkPath.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#include <wtf/Function.h>

namespace WebCore {

class GraphicsContext;
class PathStream;

class PathSkia final : public PathImpl {
public:
    static Ref<PathSkia> create();
    static Ref<PathSkia> create(const PathSegment&);
    static Ref<PathSkia> create(const PathStream&);
    static Ref<PathSkia> create(SkPath&&, RefPtr<PathStream>&&);

    PlatformPathPtr platformPath() const;

    void addPath(const PathSkia&, const AffineTransform&);

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

    bool applyElements(const PathElementApplier&) const final;

    bool transform(const AffineTransform&) final;

    bool contains(const FloatPoint&, WindRule) const;
    bool strokeContains(const FloatPoint&, const Function<void(GraphicsContext&)>& strokeStyleApplier) const;

    FloatRect strokeBoundingRect(const Function<void(GraphicsContext&)>& strokeStyleApplier) const;

private:
    PathSkia() = default;
    explicit PathSkia(const SkPath&);

    bool isEmpty() const final;

    FloatPoint currentPoint() const final;

    FloatRect fastBoundingRect() const final;
    FloatRect boundingRect() const final;

    void addEllipse(const FloatPoint&, float radiusX, float radiusY, float startAngle, float endAngle, RotationDirection);

    SkPath m_platformPath;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PathSkia)
    static bool isType(const WebCore::PathImpl& pathImpl) { return !pathImpl.isPathStream(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(SKIA)
