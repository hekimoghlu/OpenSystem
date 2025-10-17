/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#include "config.h"
#include "SVGPathByteStreamBuilder.h"

#include "SVGPathSeg.h"
#include "SVGPathStringViewSource.h"

namespace WebCore {

SVGPathByteStreamBuilder::SVGPathByteStreamBuilder(SVGPathByteStream& byteStream)
    : m_byteStream(byteStream)
{
}

void SVGPathByteStreamBuilder::moveTo(const FloatPoint& targetPoint, bool, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ?  SVGPathSegType::MoveToRel : SVGPathSegType::MoveToAbs);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::lineTo(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    writeSegmentType(mode == RelativeCoordinates ? SVGPathSegType::LineToRel : SVGPathSegType::LineToAbs);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::lineToHorizontal(float x, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::LineToHorizontalRel : SVGPathSegType::LineToHorizontalAbs);
    writeType(x);
}

void SVGPathByteStreamBuilder::lineToVertical(float y, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::LineToVerticalRel : SVGPathSegType::LineToVerticalAbs);
    writeType(y);
}

void SVGPathByteStreamBuilder::curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::CurveToCubicRel : SVGPathSegType::CurveToCubicAbs);
    writeType(point1);
    writeType(point2);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::curveToCubicSmooth(const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::CurveToCubicSmoothRel : SVGPathSegType::CurveToCubicSmoothAbs);
    writeType(point2);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::curveToQuadratic(const FloatPoint& point1, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::CurveToQuadraticRel : SVGPathSegType::CurveToQuadraticAbs);
    writeType(point1);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::curveToQuadraticSmooth(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::CurveToQuadraticSmoothRel : SVGPathSegType::CurveToQuadraticSmoothAbs);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::arcTo(float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    writeType(mode == RelativeCoordinates ? SVGPathSegType::ArcRel : SVGPathSegType::ArcAbs);
    writeType(r1);
    writeType(r2);
    writeType(angle);
    writeType(largeArcFlag);
    writeType(sweepFlag);
    writeType(targetPoint);
}

void SVGPathByteStreamBuilder::closePath()
{
    writeType(SVGPathSegType::ClosePath);
}

} // namespace WebCore
