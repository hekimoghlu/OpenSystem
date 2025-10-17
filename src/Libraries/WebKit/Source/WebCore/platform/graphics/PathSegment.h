/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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

#include "PathSegmentData.h"
#include <wtf/Function.h>

namespace WebCore {

class PathSegment {
public:
    using Data = std::variant<
        PathMoveTo,

        PathLineTo,
        PathQuadCurveTo,
        PathBezierCurveTo,
        PathArcTo,

        PathArc,
        PathClosedArc,
        PathEllipse,
        PathEllipseInRect,
        PathRect,
        PathRoundedRect,

        PathDataLine,
        PathDataQuadCurve,
        PathDataBezierCurve,
        PathDataArc,

        PathCloseSubpath
    >;

    WEBCORE_EXPORT PathSegment(Data&&);

    bool operator==(const PathSegment&) const = default;

    const Data& data() const & { return m_data; }
    Data&& data() && { return WTFMove(m_data); }
    bool closesSubpath() const { return std::holds_alternative<PathCloseSubpath>(m_data) || std::holds_alternative<PathClosedArc>(m_data); }

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;
    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    bool canApplyElements() const;
    bool applyElements(const PathElementApplier&) const;

    bool canTransform() const;
    bool transform(const AffineTransform&);

private:
    Data m_data;
};

using PathSegmentApplier = Function<void(const PathSegment&)>;

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathSegment&);

} // namespace WebCore
