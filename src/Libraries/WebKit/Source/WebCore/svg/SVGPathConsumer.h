/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#include <wtf/Noncopyable.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SVGPathConsumer;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SVGPathConsumer> : std::true_type { };
}

namespace WebCore {

enum PathCoordinateMode {
    AbsoluteCoordinates,
    RelativeCoordinates
};

enum PathParsingMode {
    NormalizedParsing,
    UnalteredParsing
};

class SVGPathConsumer : public CanMakeSingleThreadWeakPtr<SVGPathConsumer> {
    WTF_MAKE_TZONE_ALLOCATED(SVGPathConsumer);
    WTF_MAKE_NONCOPYABLE(SVGPathConsumer);
public:
    SVGPathConsumer() = default;
    virtual void incrementPathSegmentCount() = 0;
    virtual bool continueConsuming() = 0;

    // Used in UnalteredParsing/NormalizedParsing modes.
    virtual void moveTo(const FloatPoint&, bool closed, PathCoordinateMode) = 0;
    virtual void lineTo(const FloatPoint&, PathCoordinateMode) = 0;
    virtual void curveToCubic(const FloatPoint& controlPoint1, const FloatPoint& controlPoint2, const FloatPoint&, PathCoordinateMode) = 0;
    virtual void closePath() = 0;

    // Only used in UnalteredParsing mode.
    virtual void lineToHorizontal(float, PathCoordinateMode) = 0;
    virtual void lineToVertical(float, PathCoordinateMode) = 0;
    virtual void curveToCubicSmooth(const FloatPoint& controlPoint2, const FloatPoint& targetPoint, PathCoordinateMode) = 0;
    virtual void curveToQuadratic(const FloatPoint& controlPoint, const FloatPoint& targetPoint, PathCoordinateMode) = 0;
    virtual void curveToQuadraticSmooth(const FloatPoint& targetPoint, PathCoordinateMode) = 0;
    virtual void arcTo(float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag, const FloatPoint&, PathCoordinateMode) = 0;

protected:
    virtual ~SVGPathConsumer() = default;
};

} // namespace WebCore
