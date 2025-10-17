/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#include "RectangleLayoutShape.h"

#include <wtf/MathExtras.h>

namespace WebCore {

static inline float ellipseXIntercept(float y, float rx, float ry)
{
    ASSERT(ry > 0);
    return rx * sqrt(1 - (y * y) / (ry * ry));
}

FloatRect RectangleLayoutShape::shapeMarginBounds() const
{
    ASSERT(shapeMargin() >= 0);
    if (!shapeMargin())
        return m_bounds;

    float boundsX = x() - shapeMargin();
    float boundsY = y() - shapeMargin();
    float boundsWidth = width() + shapeMargin() * 2;
    float boundsHeight = height() + shapeMargin() * 2;
    return FloatRect(boundsX, boundsY, boundsWidth, boundsHeight);
}

LineSegment RectangleLayoutShape::getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const
{
    auto bounds = shapeMarginBounds();
    if (bounds.isEmpty())
        return { };

    float y1 = logicalTop;
    float y2 = logicalTop + logicalHeight;

    if (y2 < bounds.y() || y1 >= bounds.maxY())
        return { };

    float x1 = bounds.x();
    float x2 = bounds.maxX();

    float marginRadiusX = rx() + shapeMargin();
    float marginRadiusY = ry() + shapeMargin();

    if (marginRadiusY > 0) {
        if (y2 < bounds.y() + marginRadiusY) {
            float yi = y2 - bounds.y() - marginRadiusY;
            float xi = ellipseXIntercept(yi, marginRadiusX, marginRadiusY);
            x1 = bounds.x() + marginRadiusX - xi;
            x2 = bounds.maxX() - marginRadiusX + xi;
        } else if (y1 > bounds.maxY() - marginRadiusY) {
            float yi =  y1 - (bounds.maxY() - marginRadiusY);
            float xi = ellipseXIntercept(yi, marginRadiusX, marginRadiusY);
            x1 = bounds.x() + marginRadiusX - xi;
            x2 = bounds.maxX() - marginRadiusX + xi;
        }
    }

    if (writingMode().isBidiRTL())
        return { std::max(0.f, m_boxLogicalWidth - x2), std::max(0.f, m_boxLogicalWidth - x1) };
    return { x1, x2 };
}

void RectangleLayoutShape::buildDisplayPaths(DisplayPaths& paths) const
{
    paths.shape.addRoundedRect(m_bounds, FloatSize(m_radii.width(), m_radii.height()), PathRoundedRect::Strategy::PreferBezier);
    if (shapeMargin())
        paths.marginShape.addRoundedRect(shapeMarginBounds(), FloatSize(m_radii.width() + shapeMargin(), m_radii.height() + shapeMargin()), PathRoundedRect::Strategy::PreferBezier);
}

} // namespace WebCore
