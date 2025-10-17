/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#include "RenderSVGRect.h"

#include "RenderSVGShapeInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGRectElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGRect);

RenderSVGRect::RenderSVGRect(SVGRectElement& element, RenderStyle&& style)
    : RenderSVGShape(Type::SVGRect, element, WTFMove(style))
{
}

RenderSVGRect::~RenderSVGRect() = default;

SVGRectElement& RenderSVGRect::rectElement() const
{
    return downcast<SVGRectElement>(RenderSVGShape::graphicsElement());
}

void RenderSVGRect::updateShapeFromElement()
{
    // Before creating a new object we need to clear the cached bounding box
    // to avoid using garbage.
    clearPath();
    m_shapeType = ShapeType::Empty;
    m_fillBoundingBox = FloatRect();
    m_strokeBoundingBox = std::nullopt;
    m_approximateStrokeBoundingBox = std::nullopt;

    Ref rectElement = this->rectElement();
    SVGLengthContext lengthContext(rectElement.ptr());
    FloatSize boundingBoxSize(lengthContext.valueForLength(style().width(), SVGLengthMode::Width), lengthContext.valueForLength(style().height(), SVGLengthMode::Height));

    // Spec: "A negative value is illegal. A value of zero disables rendering of the element."
    if (boundingBoxSize.isEmpty())
        return;

    Ref svgStyle = style().svgStyle();
    if (lengthContext.valueForLength(svgStyle->rx(), SVGLengthMode::Width) > 0
        || lengthContext.valueForLength(svgStyle->ry(), SVGLengthMode::Height) > 0)
        m_shapeType = ShapeType::RoundedRectangle;
    else
        m_shapeType = ShapeType::Rectangle;

    if (m_shapeType != ShapeType::Rectangle || hasNonScalingStroke()) {
        // Fallback to path-based approach.
        m_fillBoundingBox = ensurePath().boundingRect();
        return;
    }

    m_fillBoundingBox = FloatRect(FloatPoint(lengthContext.valueForLength(svgStyle->x(), SVGLengthMode::Width),
        lengthContext.valueForLength(svgStyle->y(), SVGLengthMode::Height)),
        boundingBoxSize);

    auto strokeBoundingBox = m_fillBoundingBox;
    if (svgStyle->hasStroke())
        strokeBoundingBox.inflate(this->strokeWidth() / 2);

#if USE(CG)
    // CoreGraphics can inflate the stroke by 1px when drawing a rectangle with antialiasing disabled at non-integer coordinates, we need to compensate.
    if (svgStyle->shapeRendering() == ShapeRendering::CrispEdges)
        strokeBoundingBox.inflate(1);
#endif

    m_strokeBoundingBox = strokeBoundingBox;
}

void RenderSVGRect::fillShape(GraphicsContext& context) const
{
    if (hasPath()) {
        RenderSVGShape::fillShape(context);
        return;
    }

#if USE(CG)
    // FIXME: CG implementation of GraphicsContextCG::fillRect has an own
    // shadow drawing method, which draws an extra shadow.
    // This is a workaround for switching off the extra shadow.
    // https://bugs.webkit.org/show_bug.cgi?id=68899
    if (context.hasDropShadow()) {
        GraphicsContextStateSaver stateSaver(context);
        context.clearDropShadow();
        context.fillRect(m_fillBoundingBox);
        return;
    }
#endif

    context.fillRect(m_fillBoundingBox);
}

bool RenderSVGRect::canUseStrokeHitTestFastPath() const
{
    // Non-scaling-stroke needs special handling.
    if (hasNonScalingStroke())
        return false;

    // We can compute intersections with simple, continuous strokes on
    // regular rectangles without using a Path.
    return m_shapeType == ShapeType::Rectangle && definitelyHasSimpleStroke();
}

// Returns true if the stroke is continuous and definitely uses miter joins.
bool RenderSVGRect::definitelyHasSimpleStroke() const
{
    // The four angles of a rect are 90 degrees. Using the formula at:
    // http://www.w3.org/TR/SVG/painting.html#StrokeMiterlimitProperty
    // when the join style of the rect is "miter", the ratio of the miterLength
    // to the stroke-width is found to be
    // miterLength / stroke-width = 1 / sin(45 degrees)
    //                            = 1 / (1 / sqrt(2))
    //                            = sqrt(2)
    //                            = 1.414213562373095...
    // When sqrt(2) exceeds the miterlimit, then the join style switches to
    // "bevel". When the miterlimit is greater than or equal to sqrt(2) then
    // the join style remains "miter".
    //
    // An approximation of sqrt(2) is used here because at certain precise
    // miterlimits, the join style used might not be correct (e.g. a miterlimit
    // of 1.4142135 should result in bevel joins, but may be drawn using miter
    // joins).
    return style().svgStyle().strokeDashArray().isEmpty() && style().joinStyle() == LineJoin::Miter && style().strokeMiterLimit() >= 1.5;
}

void RenderSVGRect::strokeShape(GraphicsContext& context) const
{
    if (!style().hasVisibleStroke())
        return;

    if (hasPath()) {
        RenderSVGShape::strokeShape(context);
        return;
    }

    context.strokeRect(m_fillBoundingBox, strokeWidth());
}

bool RenderSVGRect::shapeDependentStrokeContains(const FloatPoint& point, PointCoordinateSpace pointCoordinateSpace)
{
    if (!canUseStrokeHitTestFastPath()) {
        ensurePath();
        return RenderSVGShape::shapeDependentStrokeContains(point, pointCoordinateSpace);
    }

    auto halfStrokeWidth = strokeWidth() / 2;
    auto halfWidth = m_fillBoundingBox.width() / 2;
    auto halfHeight = m_fillBoundingBox.height() / 2;

    auto fillBoundingBoxCenter = FloatPoint(m_fillBoundingBox.x() + halfWidth, m_fillBoundingBox.y() + halfHeight);
    auto absDeltaX = std::abs(point.x() - fillBoundingBoxCenter.x());
    auto absDeltaY = std::abs(point.y() - fillBoundingBoxCenter.y());

    if (!(absDeltaX <= halfWidth + halfStrokeWidth && absDeltaY <= halfHeight + halfStrokeWidth))
        return false;

    return (halfWidth - halfStrokeWidth <= absDeltaX) || (halfHeight - halfStrokeWidth <= absDeltaY);
}

bool RenderSVGRect::shapeDependentFillContains(const FloatPoint& point, const WindRule fillRule) const
{
    if (m_shapeType == ShapeType::Empty)
        return false;
    if (m_shapeType != ShapeType::Rectangle)
        return RenderSVGShape::shapeDependentFillContains(point, fillRule);
    return m_fillBoundingBox.contains(point.x(), point.y());
}

bool RenderSVGRect::isRenderingDisabled() const
{
    // A width or height of zero disables rendering for the element, and results in an empty bounding box.
    return m_fillBoundingBox.isEmpty();
}

}
