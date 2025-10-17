/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#include "LegacyRenderSVGEllipse.h"

#include "LegacyRenderSVGShapeInlines.h"
#include "SVGCircleElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGEllipseElement.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGEllipse);

LegacyRenderSVGEllipse::LegacyRenderSVGEllipse(SVGGraphicsElement& element, RenderStyle&& style)
    : LegacyRenderSVGShape(Type::LegacySVGEllipse, element, WTFMove(style))
{
}

LegacyRenderSVGEllipse::~LegacyRenderSVGEllipse() = default;

void LegacyRenderSVGEllipse::updateShapeFromElement()
{
    // Before creating a new object we need to clear the cached bounding box
    // to avoid using garbage.
    clearPath();
    m_shapeType = ShapeType::Empty;
    m_fillBoundingBox = FloatRect();
    m_strokeBoundingBox = std::nullopt;
    m_approximateStrokeBoundingBox = std::nullopt;
    m_center = FloatPoint();
    m_radii = FloatSize();

    calculateRadiiAndCenter();

    // Spec: "A negative value is illegal. A value of zero disables rendering of the element."
    if (m_radii.isEmpty())
        return;

    if (m_radii.width() == m_radii.height())
        m_shapeType = ShapeType::Circle;
    else
        m_shapeType = ShapeType::Ellipse;

    if (hasNonScalingStroke()) {
        // Fallback to path-based approach if shape has a non-scaling stroke.
        m_fillBoundingBox = ensurePath().boundingRect();
        return;
    }

    m_fillBoundingBox = FloatRect(m_center.x() - m_radii.width(), m_center.y() - m_radii.height(), 2 * m_radii.width(), 2 * m_radii.height());
    m_strokeBoundingBox = m_fillBoundingBox;
    if (style().svgStyle().hasStroke())
        m_strokeBoundingBox->inflate(strokeWidth() / 2);
}

void LegacyRenderSVGEllipse::calculateRadiiAndCenter()
{
    Ref graphicsElement = this->graphicsElement();
    SVGLengthContext lengthContext(graphicsElement.ptr());
    m_center = FloatPoint(
        lengthContext.valueForLength(style().svgStyle().cx(), SVGLengthMode::Width),
        lengthContext.valueForLength(style().svgStyle().cy(), SVGLengthMode::Height));
    if (is<SVGCircleElement>(graphicsElement)) {
        float radius = lengthContext.valueForLength(style().svgStyle().r());
        m_radii = FloatSize(radius, radius);
        return;
    }

    ASSERT(is<SVGEllipseElement>(graphicsElement));

    Length rx = style().svgStyle().rx();
    Length ry = style().svgStyle().ry();
    m_radii = FloatSize(
        lengthContext.valueForLength(rx.isAuto() ? ry : rx, SVGLengthMode::Width),
        lengthContext.valueForLength(ry.isAuto() ? rx : ry, SVGLengthMode::Height));
}

void LegacyRenderSVGEllipse::fillShape(GraphicsContext& context) const
{
    if (hasPath()) {
        LegacyRenderSVGShape::fillShape(context);
        return;
    }
    context.fillEllipse(m_fillBoundingBox);
}

void LegacyRenderSVGEllipse::strokeShape(GraphicsContext& context) const
{
    if (!style().hasVisibleStroke())
        return;
    if (hasPath()) {
        LegacyRenderSVGShape::strokeShape(context);
        return;
    }
    context.strokeEllipse(m_fillBoundingBox);
}

bool LegacyRenderSVGEllipse::canUseStrokeHitTestFastPath() const
{
    // Non-scaling-stroke needs special handling.
    if (hasNonScalingStroke())
        return false;

    // We can compute intersections with continuous strokes on circles
    // without using a Path.
    return m_shapeType == ShapeType::Circle && style().svgStyle().strokeDashArray().isEmpty();
}

bool LegacyRenderSVGEllipse::shapeDependentStrokeContains(const FloatPoint& point, PointCoordinateSpace pointCoordinateSpace)
{
    if (m_radii.isEmpty())
        return false;

    // The optimized code below does not support dash strokes and non-circle shape.
    // Thus we fallback to path-based approach in that case.
    if (!canUseStrokeHitTestFastPath()) {
        ensurePath();
        return LegacyRenderSVGShape::shapeDependentStrokeContains(point, pointCoordinateSpace);
    }

    float halfStrokeWidth = strokeWidth() / 2;
    FloatPoint centerOffset = FloatPoint(m_center.x() - point.x(), m_center.y() - point.y());
    return std::abs(centerOffset.length() - m_radii.width()) <= halfStrokeWidth;
}

bool LegacyRenderSVGEllipse::shapeDependentFillContains(const FloatPoint& point, const WindRule) const
{
    if (m_radii.isEmpty())
        return false;

    FloatPoint center = FloatPoint(m_center.x() - point.x(), m_center.y() - point.y());

    // This works by checking if the point satisfies the ellipse equation.
    // (x/rX)^2 + (y/rY)^2 <= 1
    float xrX = center.x() / m_radii.width();
    float yrY = center.y() / m_radii.height();
    return xrX * xrX + yrY * yrY <= 1.0;
}

bool LegacyRenderSVGEllipse::isRenderingDisabled() const
{
    // A radius of zero disables rendering of the element, and results in an empty bounding box.
    return m_fillBoundingBox.isEmpty();
}

}
