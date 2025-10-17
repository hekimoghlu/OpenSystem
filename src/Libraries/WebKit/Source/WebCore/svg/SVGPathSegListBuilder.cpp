/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#include "SVGPathSegListBuilder.h"

#include "SVGPathSegImpl.h"
#include "SVGPathSegList.h"

namespace WebCore {

SVGPathSegListBuilder::SVGPathSegListBuilder(SVGPathSegList& pathSegList)
    : m_pathSegList(pathSegList)
{
}

void SVGPathSegListBuilder::moveTo(const FloatPoint& targetPoint, bool, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegMovetoAbs::create(targetPoint.x(), targetPoint.y()));
    else
        m_pathSegList->append(SVGPathSegMovetoRel::create(targetPoint.x(), targetPoint.y()));
}

void SVGPathSegListBuilder::lineTo(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegLinetoAbs::create(targetPoint.x(), targetPoint.y()));
    else
        m_pathSegList->append(SVGPathSegLinetoRel::create(targetPoint.x(), targetPoint.y()));
}

void SVGPathSegListBuilder::lineToHorizontal(float x, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegLinetoHorizontalAbs::create(x));
    else
        m_pathSegList->append(SVGPathSegLinetoHorizontalRel::create(x));
}

void SVGPathSegListBuilder::lineToVertical(float y, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegLinetoVerticalAbs::create(y));
    else
        m_pathSegList->append(SVGPathSegLinetoVerticalRel::create(y));
}

void SVGPathSegListBuilder::curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegCurvetoCubicAbs::create(targetPoint.x(), targetPoint.y(), point1.x(), point1.y(), point2.x(), point2.y()));
    else
        m_pathSegList->append(SVGPathSegCurvetoCubicRel::create(targetPoint.x(), targetPoint.y(), point1.x(), point1.y(), point2.x(), point2.y()));
}

void SVGPathSegListBuilder::curveToCubicSmooth(const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegCurvetoCubicSmoothAbs::create(targetPoint.x(), targetPoint.y(), point2.x(), point2.y()));
    else
        m_pathSegList->append(SVGPathSegCurvetoCubicSmoothRel::create(targetPoint.x(), targetPoint.y(), point2.x(), point2.y()));
}

void SVGPathSegListBuilder::curveToQuadratic(const FloatPoint& point1, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegCurvetoQuadraticAbs::create(targetPoint.x(), targetPoint.y(), point1.x(), point1.y()));
    else
        m_pathSegList->append(SVGPathSegCurvetoQuadraticRel::create(targetPoint.x(), targetPoint.y(), point1.x(), point1.y()));
}

void SVGPathSegListBuilder::curveToQuadraticSmooth(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegCurvetoQuadraticSmoothAbs::create(targetPoint.x(), targetPoint.y()));
    else
        m_pathSegList->append(SVGPathSegCurvetoQuadraticSmoothRel::create(targetPoint.x(), targetPoint.y()));
}

void SVGPathSegListBuilder::arcTo(float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_pathSegList->append(SVGPathSegArcAbs::create(targetPoint.x(), targetPoint.y(), r1, r2, angle, largeArcFlag, sweepFlag));
    else
        m_pathSegList->append(SVGPathSegArcRel::create(targetPoint.x(), targetPoint.y(), r1, r2, angle, largeArcFlag, sweepFlag));
}

void SVGPathSegListBuilder::closePath()
{
    m_pathSegList->append(SVGPathSegClosePath::create());
}

}
