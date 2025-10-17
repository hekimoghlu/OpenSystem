/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "SVGPathAbsoluteConverter.h"

namespace WebCore {

SVGPathAbsoluteConverter::SVGPathAbsoluteConverter(SVGPathConsumer& consumer)
    : SVGPathConsumer()
    , m_consumer(consumer)
{
}

void SVGPathAbsoluteConverter::incrementPathSegmentCount()
{
    m_consumer->incrementPathSegmentCount();
}

bool SVGPathAbsoluteConverter::continueConsuming()
{
    return m_consumer->continueConsuming();
}

void SVGPathAbsoluteConverter::moveTo(const FloatPoint& targetPoint, bool closed, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->moveTo(targetPoint, closed, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->moveTo(m_currentPoint + targetPoint, closed, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }

    m_subpathPoint = m_currentPoint;
}

void SVGPathAbsoluteConverter::lineTo(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->lineTo(targetPoint, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->lineTo(m_currentPoint + targetPoint, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }
}

void SVGPathAbsoluteConverter::curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->curveToCubic(point1, point2, targetPoint, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->curveToCubic(m_currentPoint + point1, m_currentPoint + point2, m_currentPoint + targetPoint, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }
}

void SVGPathAbsoluteConverter::closePath()
{
    m_consumer->closePath();
    m_currentPoint = m_subpathPoint;
}

void SVGPathAbsoluteConverter::lineToHorizontal(float targetX, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->lineToHorizontal(targetX, AbsoluteCoordinates);
        m_currentPoint.setX(targetX);
    } else {
        auto absoluteTargetX = m_currentPoint.x() + targetX;

        m_consumer->lineToHorizontal(absoluteTargetX, AbsoluteCoordinates);
        m_currentPoint.setX(absoluteTargetX);
    }
}

void SVGPathAbsoluteConverter::lineToVertical(float targetY, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->lineToVertical(targetY, AbsoluteCoordinates);
        m_currentPoint.setY(targetY);
    } else {
        auto absoluteTargetY = m_currentPoint.y() + targetY;

        m_consumer->lineToVertical(absoluteTargetY, AbsoluteCoordinates);
        m_currentPoint.setY(absoluteTargetY);
    }
}

void SVGPathAbsoluteConverter::curveToCubicSmooth(const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->curveToCubicSmooth(point2, targetPoint, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->curveToCubicSmooth(m_currentPoint + point2, m_currentPoint + targetPoint, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }
}

void SVGPathAbsoluteConverter::curveToQuadratic(const FloatPoint& point1, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->curveToQuadratic(point1, targetPoint, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->curveToQuadratic(m_currentPoint + point1, m_currentPoint + targetPoint, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }
}

void SVGPathAbsoluteConverter::curveToQuadraticSmooth(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->curveToQuadraticSmooth(targetPoint, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->curveToQuadraticSmooth(m_currentPoint + targetPoint, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }
}

void SVGPathAbsoluteConverter::arcTo(float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates) {
        m_consumer->arcTo(r1, r2, angle, largeArcFlag, sweepFlag, targetPoint, AbsoluteCoordinates);
        m_currentPoint = targetPoint;
    } else {
        m_consumer->arcTo(r1, r2, angle, largeArcFlag, sweepFlag, m_currentPoint + targetPoint, AbsoluteCoordinates);
        m_currentPoint += targetPoint;
    }
}

} // namespace WebCore
