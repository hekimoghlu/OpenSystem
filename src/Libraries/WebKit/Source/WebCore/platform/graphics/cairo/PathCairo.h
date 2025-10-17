/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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

#if USE(CAIRO)

#include "PathImpl.h"
#include "PathStream.h"
#include "PlatformPath.h"
#include "WindRule.h"

namespace WebCore {

class GraphicsContext;
class PathStream;

class PathCairo final : public PathImpl {
public:
    static Ref<PathCairo> create();
    static Ref<PathCairo> create(const PathSegment&);
    static Ref<PathCairo> create(const PathStream&);
    static Ref<PathCairo> create(RefPtr<cairo_t>&&, RefPtr<PathStream>&& = nullptr);

    PathCairo();
    PathCairo(RefPtr<cairo_t>&&, RefPtr<PathStream>&&);

    PlatformPathPtr platformPath() const;

    void addPath(const PathCairo&, const AffineTransform&);

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
    bool isEmpty() const final;

    FloatPoint currentPoint() const final;

    FloatRect fastBoundingRect() const final;
    FloatRect boundingRect() const final;

    RefPtr<cairo_t> m_platformPath;
    RefPtr<PathStream> m_elementsStream;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PathCairo)
    static bool isType(const WebCore::PathImpl& pathImpl) { return !pathImpl.isPathStream(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(CAIRO)
