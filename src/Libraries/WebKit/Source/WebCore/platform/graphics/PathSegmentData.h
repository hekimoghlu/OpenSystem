/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

#include "FloatRect.h"
#include "FloatRoundedRect.h"
#include "PathElement.h"
#include "RotationDirection.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

struct PathMoveTo {
    FloatPoint point;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathMoveTo&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathMoveTo&);

struct PathLineTo {
    FloatPoint point;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathLineTo&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathLineTo&);

struct PathQuadCurveTo {
    FloatPoint controlPoint;
    FloatPoint endPoint;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathQuadCurveTo&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathQuadCurveTo&);

struct PathBezierCurveTo {
    FloatPoint controlPoint1;
    FloatPoint controlPoint2;
    FloatPoint endPoint;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathBezierCurveTo&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathBezierCurveTo&);

struct PathArcTo {
    FloatPoint controlPoint1;
    FloatPoint controlPoint2;
    float radius;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathArcTo&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathArcTo&);

struct PathArc {
    FloatPoint center;
    float radius;
    float startAngle;
    float endAngle;
    RotationDirection direction;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathArc&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathArc&);

struct PathClosedArc {
    PathArc arc;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathClosedArc&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathClosedArc&);

struct PathEllipse {
    FloatPoint center;
    float radiusX;
    float radiusY;
    float rotation;
    float startAngle;
    float endAngle;
    RotationDirection direction;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathEllipse&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathEllipse&);

struct PathEllipseInRect {
    FloatRect rect;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathEllipseInRect&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathEllipseInRect&);

struct PathRect {
    FloatRect rect;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathRect&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathRect&);

struct PathRoundedRect {
    enum class Strategy : uint8_t {
        PreferNative,
        PreferBezier
    };

    FloatRoundedRect roundedRect;
    Strategy strategy;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathRoundedRect&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathRoundedRect&);

struct PathDataLine {
    FloatPoint start;
    FloatPoint end;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathDataLine&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathDataLine&);

struct PathDataQuadCurve {
    FloatPoint start;
    FloatPoint controlPoint;
    FloatPoint endPoint;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathDataQuadCurve&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathDataQuadCurve&);

struct PathDataBezierCurve {
    FloatPoint start;
    FloatPoint controlPoint1;
    FloatPoint controlPoint2;
    FloatPoint endPoint;

    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathDataBezierCurve&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathDataBezierCurve&);

struct PathDataArc {
    FloatPoint start;
    FloatPoint controlPoint1;
    FloatPoint controlPoint2;
    float radius;

    static constexpr bool canApplyElements = false;
    static constexpr bool canTransform = false;

    bool operator==(const PathDataArc&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathDataArc&);

struct PathCloseSubpath {
    static constexpr bool canApplyElements = true;
    static constexpr bool canTransform = true;

    bool operator==(const PathCloseSubpath&) const = default;

    FloatPoint calculateEndPoint(const FloatPoint& currentPoint, FloatPoint& lastMoveToPoint) const;
    std::optional<FloatPoint> tryGetEndPointWithoutContext() const;

    void extendFastBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;
    void extendBoundingRect(const FloatPoint& currentPoint, const FloatPoint& lastMoveToPoint, FloatRect& boundingRect) const;

    void applyElements(const PathElementApplier&) const;

    void transform(const AffineTransform&);
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const PathCloseSubpath&);

} // namespace WebCore
