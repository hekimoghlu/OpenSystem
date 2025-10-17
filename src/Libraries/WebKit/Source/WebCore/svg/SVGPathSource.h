/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#include "SVGPathSeg.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
class SVGPathSource;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SVGPathSource> : std::true_type { };
}

namespace WebCore {

class SVGPathSource : public CanMakeSingleThreadWeakPtr<SVGPathSource> {
    WTF_MAKE_TZONE_ALLOCATED(SVGPathSource);
    WTF_MAKE_NONCOPYABLE(SVGPathSource);
public:
    SVGPathSource() = default;
    virtual ~SVGPathSource() = default;

    virtual bool hasMoreData() const = 0;
    virtual bool moveToNextToken() = 0;
    virtual SVGPathSegType nextCommand(SVGPathSegType previousCommand) = 0;

    virtual std::optional<SVGPathSegType> parseSVGSegmentType() = 0;

    struct MoveToSegment {
        FloatPoint targetPoint;
    };
    virtual std::optional<MoveToSegment> parseMoveToSegment(FloatPoint currentPoint) = 0;

    struct LineToSegment {
        FloatPoint targetPoint;
    };
    virtual std::optional<LineToSegment> parseLineToSegment(FloatPoint currentPoint) = 0;

    struct LineToHorizontalSegment {
        float x = 0;
    };
    virtual std::optional<LineToHorizontalSegment> parseLineToHorizontalSegment(FloatPoint currentPoint) = 0;

    struct LineToVerticalSegment {
        float y = 0;
    };
    virtual std::optional<LineToVerticalSegment> parseLineToVerticalSegment(FloatPoint currentPoint) = 0;

    struct CurveToCubicSegment {
        FloatPoint point1;
        FloatPoint point2;
        FloatPoint targetPoint;
    };
    virtual std::optional<CurveToCubicSegment> parseCurveToCubicSegment(FloatPoint currentPoint) = 0;

    struct CurveToCubicSmoothSegment {
        FloatPoint point2;
        FloatPoint targetPoint;
    };
    virtual std::optional<CurveToCubicSmoothSegment> parseCurveToCubicSmoothSegment(FloatPoint currentPoint) = 0;

    struct CurveToQuadraticSegment {
        FloatPoint point1;
        FloatPoint targetPoint;
    };
    virtual std::optional<CurveToQuadraticSegment> parseCurveToQuadraticSegment(FloatPoint currentPoint) = 0;

    struct CurveToQuadraticSmoothSegment {
        FloatPoint targetPoint;
    };
    virtual std::optional<CurveToQuadraticSmoothSegment> parseCurveToQuadraticSmoothSegment(FloatPoint currentPoint) = 0;

    struct ArcToSegment {
        float rx = 0;
        float ry = 0;
        float angle = 0;
        bool largeArc = false;
        bool sweep = false;
        FloatPoint targetPoint;
    };
    virtual std::optional<ArcToSegment> parseArcToSegment(FloatPoint currentPoint) = 0;
};

} // namespace WebCore
