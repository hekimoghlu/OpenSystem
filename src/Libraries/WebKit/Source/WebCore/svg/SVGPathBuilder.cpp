/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
#include "SVGPathBuilder.h"

#include "Path.h"

namespace WebCore {

SVGPathBuilder::SVGPathBuilder(Path& path)
    : m_path(path)
{
}

void SVGPathBuilder::moveTo(const FloatPoint& targetPoint, bool closed, PathCoordinateMode mode)
{
    m_current = mode == AbsoluteCoordinates ? targetPoint : m_current + targetPoint;
    if (closed && !m_path.isEmpty())
        m_path.closeSubpath();
    m_path.moveTo(m_current);
}

void SVGPathBuilder::lineTo(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    m_current = mode == AbsoluteCoordinates ? targetPoint : m_current + targetPoint;
    m_path.addLineTo(m_current);
}

void SVGPathBuilder::curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == RelativeCoordinates) {
        m_path.addBezierCurveTo(m_current + point1, m_current + point2, m_current + targetPoint);
        m_current += targetPoint;
    } else {
        m_current = targetPoint;
        m_path.addBezierCurveTo(point1, point2, m_current);
    }    
}

void SVGPathBuilder::closePath()
{
    m_path.closeSubpath();
}

}
