/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#include "RenderSVGShape.h"

#include "FloatPoint.h"
#include "FloatQuad.h"
#include "GraphicsContext.h"
#include "HitTestRequest.h"
#include "HitTestResult.h"
#include "LayoutRepainter.h"
#include "LegacyRenderSVGResourceMarker.h"
#include "PointerEventsHitRules.h"
#include "RenderSVGShapeInlines.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include "SVGPaintServerHandling.h"
#include "SVGPathData.h"
#include "SVGURIReference.h"
#include "SVGVisitedRendererTracking.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGShape);

RenderSVGShape::RenderSVGShape(Type type, SVGGraphicsElement& element, RenderStyle&& style)
    : RenderSVGModelObject(type, element, WTFMove(style), SVGModelObjectFlag::IsShape)
{
}

RenderSVGShape::~RenderSVGShape() = default;

bool RenderSVGShape::isEmpty() const
{
    // This function should never be called before assigning a new Path to m_path.
    // But this bug can happen if this renderer was created and its layout was not
    // done before painting. Assert this did not happen but do not crash.
    ASSERT(hasPath());
    return !hasPath() || path().isEmpty();
}

void RenderSVGShape::fillShape(GraphicsContext& context) const
{
    context.fillPath(path());
}

void RenderSVGShape::strokeShape(GraphicsContext& context) const
{
    ASSERT(m_path);
    Path* usePath = m_path.get();

    if (hasNonScalingStroke())
        usePath = nonScalingStrokePath(usePath, nonScalingStrokeTransform());

    context.strokePath(*usePath);
}

bool RenderSVGShape::shapeDependentStrokeContains(const FloatPoint& point, PointCoordinateSpace pointCoordinateSpace)
{
    ASSERT(m_path);

    if (hasNonScalingStroke() && pointCoordinateSpace != LocalCoordinateSpace) {
        AffineTransform nonScalingTransform = nonScalingStrokeTransform();
        Path* usePath = nonScalingStrokePath(m_path.get(), nonScalingTransform);
        return usePath->strokeContains(nonScalingTransform.mapPoint(point), [this] (GraphicsContext& context) {
            SVGRenderSupport::applyStrokeStyleToContext(context, style(), *this);
        });
    }

    return m_path->strokeContains(point, [this] (GraphicsContext& context) {
        SVGRenderSupport::applyStrokeStyleToContext(context, style(), *this);
    });
}

bool RenderSVGShape::shapeDependentFillContains(const FloatPoint& point, const WindRule fillRule) const
{
    return path().contains(point, fillRule);
}

bool RenderSVGShape::fillContains(const FloatPoint& point, bool requiresFill, const WindRule fillRule)
{
    if (m_fillBoundingBox.isEmpty() || !m_fillBoundingBox.contains(point))
        return false;

    auto paintServerResult = SVGPaintServerHandling::requestPaintServer<SVGPaintServerHandling::Operation::Fill>(*this, style());
    if (requiresFill && std::holds_alternative<std::monostate>(paintServerResult))
        return false;

    return shapeDependentFillContains(point, fillRule);
}

bool RenderSVGShape::strokeContains(const FloatPoint& point, bool requiresStroke)
{
    // "A zero value causes no stroke to be painted."
    if (!strokeWidth())
        return false;

    auto approximateStrokeBoundingBox = this->approximateStrokeBoundingBox();
    if (approximateStrokeBoundingBox.isEmpty() || !approximateStrokeBoundingBox.contains(point))
        return false;

    auto paintServerResult = SVGPaintServerHandling::requestPaintServer<SVGPaintServerHandling::Operation::Stroke>(*this, style());
    if (requiresStroke && std::holds_alternative<std::monostate>(paintServerResult))
        return false;

    return shapeDependentStrokeContains(point);
}

void RenderSVGShape::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;

    LayoutRepainter repainter(*this);
    if (m_needsShapeUpdate) {
        updateShapeFromElement();

        m_needsShapeUpdate = false;
        setCurrentSVGLayoutRect(enclosingLayoutRect(m_fillBoundingBox));
    }

    updateLayerTransform();

    repainter.repaintAfterLayout();
    clearNeedsLayout();
}

Path* RenderSVGShape::nonScalingStrokePath(const Path* path, const AffineTransform& strokeTransform) const
{
    static NeverDestroyed<Path> tempPath;

    tempPath.get() = *path;
    tempPath.get().transform(strokeTransform);

    return &tempPath.get();
}

bool RenderSVGShape::setupNonScalingStrokeContext(AffineTransform& strokeTransform, GraphicsContextStateSaver& stateSaver)
{
    std::optional<AffineTransform> inverse = strokeTransform.inverse();
    if (!inverse)
        return false;

    stateSaver.save();
    stateSaver.context()->concatCTM(inverse.value());
    return true;
}

AffineTransform RenderSVGShape::nonScalingStrokeTransform() const
{
    return protectedGraphicsElement()->getScreenCTM(SVGLocatable::DisallowStyleUpdate);
}

void RenderSVGShape::fillShape(const RenderStyle& style, GraphicsContext& context)
{
    SVGPaintServerHandling paintServerHandling { context };
    if (paintServerHandling.preparePaintOperation<SVGPaintServerHandling::Operation::Fill>(*this, style))
        fillShape(context);
}

void RenderSVGShape::strokeShape(const RenderStyle& style, GraphicsContext& context)
{
    if (!style.hasVisibleStroke())
        return;

    GraphicsContextStateSaver stateSaver(context, false);
    if (hasNonScalingStroke()) {
        AffineTransform nonScalingTransform = nonScalingStrokeTransform();
        if (!setupNonScalingStrokeContext(nonScalingTransform, stateSaver))
            return;
    }

    SVGPaintServerHandling paintServerHandling { context };
    if (paintServerHandling.preparePaintOperation<SVGPaintServerHandling::Operation::Stroke>(*this, style))
        strokeShape(context);
}

void RenderSVGShape::fillStrokeMarkers(PaintInfo& childPaintInfo)
{
    for (auto type : RenderStyle::paintTypesForPaintOrder(style().paintOrder())) {
        switch (type) {
        case PaintType::Fill:
            fillShape(style(), childPaintInfo.context());
            break;
        case PaintType::Stroke:
            strokeShape(style(), childPaintInfo.context());
            break;
        case PaintType::Markers:
            drawMarkers(childPaintInfo);
            break;
        }
    }
}

void RenderSVGShape::paint(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    OptionSet<PaintPhase> relevantPaintPhases { PaintPhase::Foreground, PaintPhase::ClippingMask, PaintPhase::Mask, PaintPhase::Outline, PaintPhase::SelfOutline };
    if (!shouldPaintSVGRenderer(paintInfo, relevantPaintPhases) || isEmpty())
        return;

    if (paintInfo.phase == PaintPhase::ClippingMask) {
        paintSVGClippingMask(paintInfo, objectBoundingBox());
        return;
    }

    auto adjustedPaintOffset = paintOffset + currentSVGLayoutLocation();
    if (paintInfo.phase == PaintPhase::Mask) {
        paintSVGMask(paintInfo, adjustedPaintOffset);
        return;
    }

    auto visualOverflowRect = visualOverflowRectEquivalent();
    visualOverflowRect.moveBy(adjustedPaintOffset);
    if (!visualOverflowRect.intersects(paintInfo.rect))
        return;

    if (paintInfo.phase == PaintPhase::Outline || paintInfo.phase == PaintPhase::SelfOutline) {
        paintSVGOutline(paintInfo, adjustedPaintOffset);
        return;
    }

    ASSERT(paintInfo.phase == PaintPhase::Foreground);
    GraphicsContextStateSaver stateSaver(paintInfo.context());

    auto coordinateSystemOriginTranslation = adjustedPaintOffset - nominalSVGLayoutLocation();
    paintInfo.context().translate(coordinateSystemOriginTranslation.width(), coordinateSystemOriginTranslation.height());

    if (style().svgStyle().shapeRendering() == ShapeRendering::CrispEdges)
        paintInfo.context().setShouldAntialias(false);

    fillStrokeMarkers(paintInfo);
}

bool RenderSVGShape::isPointInFill(const FloatPoint& point)
{
    return shapeDependentFillContains(point, style().svgStyle().fillRule());
}

bool RenderSVGShape::isPointInStroke(const FloatPoint& point)
{
    if (!style().svgStyle().hasStroke())
        return false;

    return shapeDependentStrokeContains(point, LocalCoordinateSpace);
}

float RenderSVGShape::getTotalLength() const
{
    return hasPath() ? path().length() : createPath()->length();
}

FloatPoint RenderSVGShape::getPointAtLength(float distance) const
{
    return hasPath() ? path().pointAtLength(distance) : createPath()->pointAtLength(distance);
}

bool RenderSVGShape::nodeAtPoint(const HitTestRequest& request, HitTestResult& result, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction hitTestAction)
{
    if (hitTestAction != HitTestForeground)
        return false;

    static NeverDestroyed<SVGVisitedRendererTracking::VisitedSet> s_visitedSet;

    SVGVisitedRendererTracking recursionTracking(s_visitedSet);
    if (recursionTracking.isVisiting(*this))
        return false;

    SVGVisitedRendererTracking::Scope recursionScope(recursionTracking, *this);

    auto adjustedLocation = accumulatedOffset + currentSVGLayoutLocation();
    auto localPoint = locationInContainer.point();
    auto coordinateSystemOriginTranslation = nominalSVGLayoutLocation() - adjustedLocation;
    localPoint.move(coordinateSystemOriginTranslation);

    if (!pointInSVGClippingArea(localPoint))
        return false;

    PointerEventsHitRules hitRules(PointerEventsHitRules::HitTestingTargetType::SVGPath, request, usedPointerEvents());
    if (isVisibleToHitTesting(style(), request) || !hitRules.requireVisible) {
        const SVGRenderStyle& svgStyle = style().svgStyle();
        WindRule fillRule = svgStyle.fillRule();
        if (request.svgClipContent())
            fillRule = svgStyle.clipRule();

        if (hitRules.canHitStroke && (svgStyle.hasStroke() || !hitRules.requireStroke) && strokeContains(localPoint, hitRules.requireStroke)) {
            updateHitTestResult(result, locationInContainer.point() - toLayoutSize(adjustedLocation));
            if (result.addNodeToListBasedTestResult(protectedNodeForHitTest().get(), request, locationInContainer, strokeBoundingBox()) == HitTestProgress::Stop)
                return true;
            return false;
        }

        if ((hitRules.canHitFill && (svgStyle.hasFill() || !hitRules.requireFill) && fillContains(localPoint, hitRules.requireFill, fillRule))
            || (hitRules.canHitBoundingBox && m_fillBoundingBox.contains(localPoint))) {
            updateHitTestResult(result, locationInContainer.point() - toLayoutSize(adjustedLocation));
            if (result.addNodeToListBasedTestResult(protectedNodeForHitTest().get(), request, locationInContainer, m_fillBoundingBox) == HitTestProgress::Stop)
                return true;
            return false;
        }
    }

    return false;
}

FloatRect RenderSVGShape::strokeBoundingBox() const
{
    if (m_shapeType == ShapeType::Empty)
        return { };
    if (!m_strokeBoundingBox) {
        // Initialize m_strokeBoundingBox before calling calculateStrokeBoundingBox, since recursively referenced markers can cause us to re-enter here.
        m_strokeBoundingBox = FloatRect { };
        m_strokeBoundingBox = calculateStrokeBoundingBox();
    }
    return *m_strokeBoundingBox;
}

FloatRect RenderSVGShape::calculateStrokeBoundingBox() const
{
    ASSERT(m_path);
    FloatRect strokeBoundingBox = m_fillBoundingBox;

    if (style().svgStyle().hasStroke()) {
        if (hasNonScalingStroke()) {
            AffineTransform nonScalingTransform = nonScalingStrokeTransform();
            if (std::optional<AffineTransform> inverse = nonScalingTransform.inverse()) {
                Path* usePath = nonScalingStrokePath(m_path.get(), nonScalingTransform);
                FloatRect strokeBoundingRect = usePath->strokeBoundingRect(Function<void(GraphicsContext&)> { [this] (GraphicsContext& context) {
                    SVGRenderSupport::applyStrokeStyleToContext(context, style(), *this);
                } });
                strokeBoundingRect = inverse.value().mapRect(strokeBoundingRect);
                strokeBoundingBox.unite(strokeBoundingRect);
            }
        } else {
            strokeBoundingBox.unite(path().strokeBoundingRect(Function<void(GraphicsContext&)> { [this] (GraphicsContext& context) {
                SVGRenderSupport::applyStrokeStyleToContext(context, style(), *this);
            } }));
        }
    }

    return adjustStrokeBoundingBoxForZeroLengthLinecaps(RepaintRectCalculation::Accurate, strokeBoundingBox);
}

FloatRect RenderSVGShape::approximateStrokeBoundingBox() const
{
    if (m_shapeType == ShapeType::Empty)
        return { };
    if (!m_approximateStrokeBoundingBox) {
        // Initialize m_approximateStrokeBoundingBox before calling calculateApproximateStrokeBoundingBox, since recursively referenced markers can cause us to re-enter here.
        m_approximateStrokeBoundingBox = FloatRect { };
        m_approximateStrokeBoundingBox = calculateApproximateStrokeBoundingBox();
    }
    return *m_approximateStrokeBoundingBox;
}

FloatRect RenderSVGShape::calculateApproximateStrokeBoundingBox() const
{
    if (m_strokeBoundingBox)
        return *m_strokeBoundingBox;

    return SVGRenderSupport::calculateApproximateStrokeBoundingBox(*this);
}

float RenderSVGShape::strokeWidth() const
{
    SVGLengthContext lengthContext(protectedGraphicsElement().ptr());
    auto strokeWidth = lengthContext.valueForLength(style().strokeWidth());
    return std::isnan(strokeWidth) ? 0 : strokeWidth;
}

Path& RenderSVGShape::ensurePath()
{
    if (!hasPath())
        m_path = createPath();
    return path();
}

std::unique_ptr<Path> RenderSVGShape::createPath() const
{
    return makeUnique<Path>(pathFromGraphicsElement(protectedGraphicsElement()));
}

void RenderSVGShape::styleWillChange(StyleDifference diff, const RenderStyle& newStyle)
{
    auto* oldStyle = hasInitializedStyle() ? &style() : nullptr;
    if (oldStyle) {
        if (diff == StyleDifference::Layout)
            setNeedsShapeUpdate();
    }

    RenderSVGModelObject::styleWillChange(diff, newStyle);
}

bool RenderSVGShape::needsHasSVGTransformFlags() const
{
    return protectedGraphicsElement()->hasTransformRelatedAttributes();
}

void RenderSVGShape::applyTransform(TransformationMatrix& transform, const RenderStyle& style, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption> options) const
{
    applySVGTransform(transform, protectedGraphicsElement(), style, boundingBox, std::nullopt, std::nullopt, options);
}

}
