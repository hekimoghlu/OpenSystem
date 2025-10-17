/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#include "SVGPathByteStreamSource.h"

namespace WebCore {

SVGPathByteStreamSource::SVGPathByteStreamSource(const SVGPathByteStream& stream)
    : m_streamCurrent(stream.bytes().span())
{
}

bool SVGPathByteStreamSource::hasMoreData() const
{
    return !m_streamCurrent.empty();
}

SVGPathSegType SVGPathByteStreamSource::nextCommand(SVGPathSegType)
{
    return readSVGSegmentType();
}

std::optional<SVGPathSegType> SVGPathByteStreamSource::parseSVGSegmentType()
{
    return readSVGSegmentType();
}

std::optional<SVGPathSource::MoveToSegment> SVGPathByteStreamSource::parseMoveToSegment(FloatPoint)
{
    MoveToSegment segment;
    segment.targetPoint = readFloatPoint();
    return segment;
}

std::optional<SVGPathSource::LineToSegment> SVGPathByteStreamSource::parseLineToSegment(FloatPoint)
{
    LineToSegment segment;
    segment.targetPoint = readFloatPoint();
    return segment;
}

std::optional<SVGPathSource::LineToHorizontalSegment> SVGPathByteStreamSource::parseLineToHorizontalSegment(FloatPoint)
{
    LineToHorizontalSegment segment;
    segment.x = readFloat();
    return segment;
}

std::optional<SVGPathSource::LineToVerticalSegment> SVGPathByteStreamSource::parseLineToVerticalSegment(FloatPoint)
{
    LineToVerticalSegment segment;
    segment.y = readFloat();
    return segment;
}

std::optional<SVGPathSource::CurveToCubicSegment> SVGPathByteStreamSource::parseCurveToCubicSegment(FloatPoint)
{
    CurveToCubicSegment segment;
    segment.point1 = readFloatPoint();
    segment.point2 = readFloatPoint();
    segment.targetPoint = readFloatPoint();
    return segment;
}

std::optional<SVGPathSource::CurveToCubicSmoothSegment> SVGPathByteStreamSource::parseCurveToCubicSmoothSegment(FloatPoint)
{
    CurveToCubicSmoothSegment segment;
    segment.point2 = readFloatPoint();
    segment.targetPoint = readFloatPoint();
    return segment;
}

std::optional<SVGPathSource::CurveToQuadraticSegment> SVGPathByteStreamSource::parseCurveToQuadraticSegment(FloatPoint)
{
    CurveToQuadraticSegment segment;
    segment.point1 = readFloatPoint();
    segment.targetPoint = readFloatPoint();
    return segment;
}

std::optional<SVGPathSource::CurveToQuadraticSmoothSegment> SVGPathByteStreamSource::parseCurveToQuadraticSmoothSegment(FloatPoint)
{
    CurveToQuadraticSmoothSegment segment;
    segment.targetPoint = readFloatPoint();
    return segment;
}

std::optional<SVGPathSource::ArcToSegment> SVGPathByteStreamSource::parseArcToSegment(FloatPoint)
{
    ArcToSegment segment;
    segment.rx = readFloat();
    segment.ry = readFloat();
    segment.angle = readFloat();
    segment.largeArc = readFlag();
    segment.sweep = readFlag();
    segment.targetPoint = readFloatPoint();
    return segment;
}

}
