/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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

#if USE(CG)

#include "PathImpl.h"
#include "PlatformPath.h"
#include "WindRule.h"
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>

typedef struct CGContext* CGContextRef;

namespace WebCore {

class GraphicsContext;
class Path;
class PathStream;

class PathCG final : public PathImpl {
    WTF_MAKE_TZONE_ALLOCATED(PathCG);
public:
    static Ref<PathCG> create();
    static Ref<PathCG> create(const PathSegment&);
    static Ref<PathCG> create(const PathStream&);
    static Ref<PathCG> create(RetainPtr<CGMutablePathRef>&&);

    PlatformPathPtr platformPath() const;

    void addPath(const PathCG&, const AffineTransform&);

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
    PathCG();
    PathCG(RetainPtr<CGMutablePathRef>&&);

    PlatformPathPtr ensureMutablePlatformPath();

    bool isEmpty() const final;

    FloatPoint currentPoint() const final;

    FloatRect fastBoundingRect() const final;
    FloatRect boundingRect() const final;

    RetainPtr<CGMutablePathRef> m_platformPath;
};

void addToCGContextPath(CGContextRef, const Path&);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PathCG)
    static bool isType(const WebCore::PathImpl& pathImpl) { return !pathImpl.isPathStream(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(CG)
