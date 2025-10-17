/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#include "RenderSVGModelObject.h"

#include "RenderElementInlines.h"
#include "RenderGeometryMap.h"
#include "RenderLayer.h"
#include "RenderLayerInlines.h"
#include "RenderLayerModelObject.h"
#include "RenderObjectInlines.h"
#include "RenderSVGModelObjectInlines.h"
#include "RenderView.h"
#include "SVGElementInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGraphicsElement.h"
#include "SVGLocatable.h"
#include "SVGNames.h"
#include "SVGPathData.h"
#include "SVGUseElement.h"
#include "TransformState.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGModelObject);

RenderSVGModelObject::RenderSVGModelObject(Type type, Document& document, RenderStyle&& style, OptionSet<SVGModelObjectFlag> typeFlags)
    : RenderLayerModelObject(type, document, WTFMove(style), { }, typeFlags)
{
    ASSERT(!isLegacyRenderSVGModelObject());
    ASSERT(isRenderSVGModelObject());
}

RenderSVGModelObject::RenderSVGModelObject(Type type, SVGElement& element, RenderStyle&& style, OptionSet<SVGModelObjectFlag> typeFlags)
    : RenderLayerModelObject(type, element, WTFMove(style), { }, typeFlags)
{
    ASSERT(!isLegacyRenderSVGModelObject());
    ASSERT(isRenderSVGModelObject());
}

RenderSVGModelObject::~RenderSVGModelObject() = default;

void RenderSVGModelObject::updateFromStyle()
{
    RenderLayerModelObject::updateFromStyle();
    updateHasSVGTransformFlags();
}

LayoutRect RenderSVGModelObject::overflowClipRect(const LayoutPoint&, OverlayScrollbarSizeRelevancy, PaintPhase) const
{
    ASSERT_NOT_REACHED();
    return LayoutRect();
}

auto RenderSVGModelObject::localRectsForRepaint(RepaintOutlineBounds repaintOutlineBounds) const -> RepaintRects
{
    if (isInsideEntirelyHiddenLayer())
        return { };

    ASSERT(!view().frameView().layoutContext().isPaintOffsetCacheEnabled());

    auto visualOverflowRect = visualOverflowRectEquivalent();
    auto rects = RepaintRects { visualOverflowRect };
    if (repaintOutlineBounds == RepaintOutlineBounds::Yes)
        rects.outlineBoundsRect = visualOverflowRect;

    return rects;
}

auto RenderSVGModelObject::computeVisibleRectsInContainer(const RepaintRects& rects, const RenderLayerModelObject* container, VisibleRectContext context) const -> std::optional<RepaintRects>
{
    return computeVisibleRectsInSVGContainer(rects, container, context);
}

const RenderObject* RenderSVGModelObject::pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap& geometryMap) const
{
    ASSERT(ancestorToStopAt != this);
    ASSERT(style().position() == PositionType::Static);

    bool ancestorSkipped;
    CheckedPtr container = this->container(ancestorToStopAt, ancestorSkipped);
    if (!container)
        return nullptr;

    ASSERT_UNUSED(ancestorSkipped, !ancestorSkipped);

    pushOntoGeometryMap(geometryMap, ancestorToStopAt, container.get(), ancestorSkipped);
    return container.get();
}

LayoutRect RenderSVGModelObject::outlineBoundsForRepaint(const RenderLayerModelObject* repaintContainer, const RenderGeometryMap* geometryMap) const
{
    ASSERT(!view().frameView().layoutContext().isPaintOffsetCacheEnabled());

    auto outlineBounds = visualOverflowRectEquivalent();

    if (repaintContainer != this) {
        FloatQuad containerRelativeQuad;
        if (geometryMap)
            containerRelativeQuad = geometryMap->mapToContainer(outlineBounds, repaintContainer);
        else
            containerRelativeQuad = localToContainerQuad(FloatRect(outlineBounds), repaintContainer);

        outlineBounds = LayoutRect(containerRelativeQuad.boundingBox());
    }

    return outlineBounds;
}

void RenderSVGModelObject::boundingRects(Vector<LayoutRect>& rects, const LayoutPoint& accumulatedOffset) const
{
    rects.append({ accumulatedOffset, m_layoutRect.size() });
}

void RenderSVGModelObject::absoluteQuads(Vector<FloatQuad>& quads, bool* wasFixed) const
{
    quads.append(localToAbsoluteQuad(FloatRect { { }, m_layoutRect.size() }, UseTransforms, wasFixed));
}

void RenderSVGModelObject::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderLayerModelObject::styleDidChange(diff, oldStyle);

    // SVG masks are painted independent of the target renderers visibility.
    // FIXME: [LBSE] Upstream RenderElement changes
    // bool hasSVGMask = hasSVGMask();
    bool hasSVGMask = false;
    if (hasSVGMask && hasLayer() && style().usedVisibility() != Visibility::Visible)
        layer()->setHasVisibleContent();
}

void RenderSVGModelObject::mapAbsoluteToLocalPoint(OptionSet<MapCoordinatesMode> mode, TransformState& transformState) const
{
    ASSERT(style().position() == PositionType::Static);

    if (isTransformed())
        mode.remove(IsFixed);

    auto* container = parent();
    if (!container)
        return;

    container->mapAbsoluteToLocalPoint(mode, transformState);

    LayoutSize containerOffset = offsetFromContainer(*container, LayoutPoint());

    pushOntoTransformState(transformState, mode, nullptr, container, containerOffset, false);
}

void RenderSVGModelObject::mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState& transformState, OptionSet<MapCoordinatesMode> mode, bool* wasFixed) const
{
    mapLocalToSVGContainer(ancestorContainer, transformState, mode, wasFixed);
}

LayoutSize RenderSVGModelObject::offsetFromContainer(RenderElement& container, const LayoutPoint&, bool*) const
{
    ASSERT_UNUSED(container, &container == this->container());
    ASSERT(!isInFlowPositioned());
    ASSERT(!isAbsolutelyPositioned());
    ASSERT(isInline());
    return locationOffsetEquivalent();
}

void RenderSVGModelObject::addFocusRingRects(Vector<LayoutRect>& rects, const LayoutPoint& additionalOffset, const RenderLayerModelObject*) const
{
    auto repaintBoundingBox = enclosingLayoutRect(repaintRectInLocalCoordinates());
    if (repaintBoundingBox.size().isEmpty())
        return;
    rects.append(LayoutRect(additionalOffset, repaintBoundingBox.size()));
}

// FloatRect::intersects does not consider horizontal or vertical lines (because of isEmpty()).
// So special-case handling of such lines.
static bool intersectsAllowingEmpty(const FloatRect& r, const FloatRect& other)
{
    if (r.isEmpty() && other.isEmpty())
        return false;
    if (r.isEmpty() && !other.isEmpty())
        return (other.contains(r.x(), r.y()) && !other.contains(r.maxX(), r.maxY())) || (!other.contains(r.x(), r.y()) && other.contains(r.maxX(), r.maxY()));
    if (other.isEmpty())
        return intersectsAllowingEmpty(other, r);
    return r.intersects(other);
}

// One of the element types that can cause graphics to be drawn onto the target canvas. Specifically: circle, ellipse,
// image, line, path, polygon, polyline, rect, text and use.
static bool isGraphicsElement(const RenderElement& renderer)
{
    return renderer.isRenderSVGShape() || renderer.isRenderSVGText() || renderer.isRenderSVGImage() || renderer.element()->hasTagName(SVGNames::useTag);
}

bool RenderSVGModelObject::checkIntersection(RenderElement* renderer, const FloatRect& rect)
{
    if (!renderer || renderer->usedPointerEvents() == PointerEvents::None)
        return false;
    if (!isGraphicsElement(*renderer))
        return false;
    RefPtr svgElement = downcast<SVGGraphicsElement>(renderer->element());
    auto ctm = svgElement->getCTM(SVGLocatable::DisallowStyleUpdate);
    // FIXME: [SVG] checkEnclosure implementation is inconsistent
    // https://bugs.webkit.org/show_bug.cgi?id=262709
    return intersectsAllowingEmpty(rect, ctm.mapRect(renderer->repaintRectInLocalCoordinates(RepaintRectCalculation::Accurate)));
}

bool RenderSVGModelObject::checkEnclosure(RenderElement* renderer, const FloatRect& rect)
{
    if (!renderer || renderer->usedPointerEvents() == PointerEvents::None)
        return false;
    if (!isGraphicsElement(*renderer))
        return false;
    RefPtr svgElement = downcast<SVGGraphicsElement>(renderer->element());
    auto ctm = svgElement->getCTM(SVGLocatable::DisallowStyleUpdate);
    // FIXME: [SVG] checkEnclosure implementation is inconsistent
    // https://bugs.webkit.org/show_bug.cgi?id=262709
    return rect.contains(ctm.mapRect(renderer->repaintRectInLocalCoordinates(RepaintRectCalculation::Accurate)));
}

LayoutSize RenderSVGModelObject::cachedSizeForOverflowClip() const
{
    ASSERT(hasNonVisibleOverflow());
    ASSERT(hasLayer());
    return layer()->size();
}

bool RenderSVGModelObject::applyCachedClipAndScrollPosition(RepaintRects& rects, const RenderLayerModelObject* container, VisibleRectContext context) const
{
    // Based on RenderBox::applyCachedClipAndScrollPosition -- unused options removed.
    if (!context.options.contains(VisibleRectContextOption::ApplyContainerClip) && this == container)
        return true;

    LayoutRect clipRect(LayoutPoint(), cachedSizeForOverflowClip());
    if (effectiveOverflowX() == Overflow::Visible)
        clipRect.expandToInfiniteX();
    if (effectiveOverflowY() == Overflow::Visible)
        clipRect.expandToInfiniteY();

    bool intersects;
    if (context.options.contains(VisibleRectContextOption::UseEdgeInclusiveIntersection))
        intersects = rects.edgeInclusiveIntersect(clipRect);
    else
        intersects = rects.intersect(clipRect);

    return intersects;
}

Path RenderSVGModelObject::computeClipPath(AffineTransform& transform) const
{
    if (layer()->isTransformed())
        transform.multiply(layer()->currentTransform(RenderStyle::individualTransformOperations()).toAffineTransform());

    if (RefPtr useElement = dynamicDowncast<SVGUseElement>(protectedElement())) {
        if (CheckedPtr clipChildRenderer = useElement->rendererClipChild())
            transform.multiply(downcast<RenderLayerModelObject>(*clipChildRenderer).checkedLayer()->currentTransform(RenderStyle::individualTransformOperations()).toAffineTransform());
        if (RefPtr clipChild = useElement->clipChild())
            return pathFromGraphicsElement(*clipChild);
    }

    return pathFromGraphicsElement(Ref { downcast<SVGGraphicsElement>(element()) });
}

void RenderSVGModelObject::paintSVGOutline(PaintInfo& paintInfo, const LayoutPoint& adjustedPaintOffset)
{
    paintOutline(paintInfo, LayoutRect(adjustedPaintOffset, borderBoxRectEquivalent().size()));
}

} // namespace WebCore
