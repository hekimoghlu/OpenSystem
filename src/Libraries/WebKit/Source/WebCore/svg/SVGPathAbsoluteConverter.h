/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

#include "SVGPathConsumer.h"

namespace WebCore {

// This consumer translates incoming SVG draw commands into absolute coordinates, and forwards it
// to another consumer. convertSVGPathByteStreamToAbsoluteCoordinates uses it to convert relative
// draw commands in an SVG path into absolute.
class SVGPathAbsoluteConverter final : public SVGPathConsumer {
public:
    SVGPathAbsoluteConverter(SVGPathConsumer&);

private:
    void incrementPathSegmentCount() final;
    bool continueConsuming() final;

    // Used in UnalteredParsing/NormalizedParsing modes.
    void moveTo(const FloatPoint& targetPoint, bool closed, PathCoordinateMode) final;
    void lineTo(const FloatPoint& targetPoint, PathCoordinateMode) final;
    void curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode) final;
    void closePath() final;

    // Only used in UnalteredParsing mode.
    void lineToHorizontal(float targetX, PathCoordinateMode) final;
    void lineToVertical(float targetY, PathCoordinateMode) final;
    void curveToCubicSmooth(const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode) final;
    void curveToQuadratic(const FloatPoint& point1, const FloatPoint& targetPoint, PathCoordinateMode) final;
    void curveToQuadraticSmooth(const FloatPoint& targetPoint, PathCoordinateMode) final;
    void arcTo(float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag, const FloatPoint& targetPoint, PathCoordinateMode) final;

    SingleThreadWeakRef<SVGPathConsumer> m_consumer;
    FloatPoint m_currentPoint;
    FloatPoint m_subpathPoint;
};

} // namespace WebCore
