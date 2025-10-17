/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#include "LegacyRenderSVGModelObject.h"

#include "LegacyRenderSVGResource.h"
#include "NotImplemented.h"
#include "RenderLayer.h"
#include "RenderLayerModelObject.h"
#include "RenderView.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include "SVGResourcesCache.h"
#include "ShadowRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGModelObject);

LegacyRenderSVGModelObject::LegacyRenderSVGModelObject(Type type, SVGElement& element, RenderStyle&& style, OptionSet<SVGModelObjectFlag> typeFlags)
    : RenderElement(type, element, WTFMove(style), { }, typeFlags | SVGModelObjectFlag::IsLegacy | SVGModelObjectFlag::UsesBoundaryCaching)
{
    ASSERT(isLegacyRenderSVGModelObject());
    ASSERT(!isRenderSVGModelObject());
}

LegacyRenderSVGModelObject::~LegacyRenderSVGModelObject() = default;

LayoutRect LegacyRenderSVGModelObject::clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext context) const
{
    return SVGRenderSupport::clippedOverflowRectForRepaint(*this, repaintContainer, context);
}

auto LegacyRenderSVGModelObject::rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds repaintOutlineBounds) const -> RepaintRects
{
    auto rects = RepaintRects { clippedOverflowRect(repaintContainer, visibleRectContextForRepaint()) };
    if (repaintOutlineBounds == RepaintOutlineBounds::Yes)
        rects.outlineBoundsRect = outlineBoundsForRepaint(repaintContainer);

    return rects;
}

std::optional<FloatRect> LegacyRenderSVGModelObject::computeFloatVisibleRectInContainer(const FloatRect& rect, const RenderLayerModelObject* container, VisibleRectContext context) const
{
    return SVGRenderSupport::computeFloatVisibleRectInContainer(*this, rect, container, context);
}

void LegacyRenderSVGModelObject::mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState& transformState, OptionSet<MapCoordinatesMode>, bool* wasFixed) const
{
    SVGRenderSupport::mapLocalToContainer(*this, ancestorContainer, transformState, wasFixed);
}

const RenderObject* LegacyRenderSVGModelObject::pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap& geometryMap) const
{
    return SVGRenderSupport::pushMappingToContainer(*this, ancestorToStopAt, geometryMap);
}

static void adjustRectForOutlineAndShadow(const RenderObject& renderer, LayoutRect& rect)
{
    auto shadowRect = rect;
    if (auto* boxShadow = renderer.style().boxShadow())
        boxShadow->adjustRectForShadow(shadowRect);

    auto outlineRect = rect;
    auto outlineSize = LayoutUnit { renderer.outlineStyleForRepaint().outlineSize() };
    if (outlineSize)
        outlineRect.inflate(outlineSize);

    rect = unionRect(shadowRect, outlineRect);
}

// Copied from RenderBox, this method likely requires further refactoring to work easily for both SVG and CSS Box Model content.
// FIXME: This may also need to move into SVGRenderSupport as the RenderBox version depends
// on borderBoundingBox() which SVG RenderBox subclases (like SVGRenderBlock) do not implement.
LayoutRect LegacyRenderSVGModelObject::outlineBoundsForRepaint(const RenderLayerModelObject* repaintContainer, const RenderGeometryMap*) const
{
    LayoutRect box = enclosingLayoutRect(repaintRectInLocalCoordinates());
    adjustRectForOutlineAndShadow(*this, box);

    FloatQuad containerRelativeQuad = localToContainerQuad(FloatRect(box), repaintContainer);
    return LayoutRect(snapRectToDevicePixels(LayoutRect(containerRelativeQuad.boundingBox()), document().deviceScaleFactor()));
}

void LegacyRenderSVGModelObject::boundingRects(Vector<LayoutRect>& rects, const LayoutPoint& accumulatedOffset) const
{
    auto rect = LayoutRect { strokeBoundingBox() };
    rect.moveBy(accumulatedOffset);
    rects.append(rect);
}

void LegacyRenderSVGModelObject::absoluteQuads(Vector<FloatQuad>& quads, bool* wasFixed) const
{
    quads.append(localToAbsoluteQuad(strokeBoundingBox(), UseTransforms, wasFixed));
}

void LegacyRenderSVGModelObject::willBeDestroyed()
{
    SVGResourcesCache::clientDestroyed(*this);
    RenderElement::willBeDestroyed();
}

void LegacyRenderSVGModelObject::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    if (diff == StyleDifference::Layout) {
        invalidateCachedBoundaries();
        if (style().affectsTransform() || (oldStyle && oldStyle->affectsTransform()))
            setNeedsTransformUpdate();
    }
    RenderElement::styleDidChange(diff, oldStyle);
    SVGResourcesCache::clientStyleChanged(*this, diff, oldStyle, style());
}

bool LegacyRenderSVGModelObject::nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation&, const LayoutPoint&, HitTestAction)
{
    ASSERT_NOT_REACHED();
    return false;
}

static void getElementCTM(SVGElement* element, AffineTransform& transform)
{
    ASSERT(element);

    RefPtr stopAtElement = SVGLocatable::nearestViewportElement(element);
    ASSERT(stopAtElement);

    AffineTransform localTransform;
    Node* current = element;

    while (RefPtr currentElement = dynamicDowncast<SVGElement>(current)) {
        localTransform = currentElement->renderer()->localToParentTransform();
        transform = localTransform.multiply(transform);
        // For getCTM() computation, stop at the nearest viewport element
        if (currentElement == stopAtElement.get())
            break;

        current = current->parentOrShadowHostNode();
    }
}

// FloatRect::intersects does not consider horizontal or vertical lines (because of isEmpty()).
// So special-case handling of such lines.
static bool intersectsAllowingEmpty(const FloatRect& r, const FloatRect& other)
{
    if (r.isEmpty() && other.isEmpty())
        return false;
    if (r.isEmpty() && !other.isEmpty())
        return (other.contains(r.x(), r.y()) && !other.contains(r.maxX(), r.maxY())) || (!other.contains(r.x(), r.y()) && other.contains(r.maxX(), r.maxY()));
    if (other.isEmpty() && !r.isEmpty())
        return intersectsAllowingEmpty(other, r);
    return r.intersects(other);
}

// One of the element types that can cause graphics to be drawn onto the target canvas. Specifically: circle, ellipse,
// image, line, path, polygon, polyline, rect, text and use.
static bool isGraphicsElement(const RenderElement& renderer)
{
    return renderer.isLegacyRenderSVGShape() || renderer.isRenderSVGText() || renderer.isLegacyRenderSVGImage() || renderer.element()->hasTagName(SVGNames::useTag);
}

// The SVG addFocusRingRects() method adds rects in local coordinates so the default absoluteFocusRingQuads
// returns incorrect values for SVG objects. Overriding this method provides access to the absolute bounds.
void LegacyRenderSVGModelObject::absoluteFocusRingQuads(Vector<FloatQuad>& quads)
{
    quads.append(localToAbsoluteQuad(FloatQuad(repaintRectInLocalCoordinates())));
}
    
bool LegacyRenderSVGModelObject::checkIntersection(RenderElement* renderer, const FloatRect& rect)
{
    if (!renderer || renderer->usedPointerEvents() == PointerEvents::None)
        return false;
    if (!isGraphicsElement(*renderer))
        return false;
    AffineTransform ctm;
    RefPtr svgElement = downcast<SVGElement>(renderer->element());
    getElementCTM(svgElement.get(), ctm);
    ASSERT(svgElement->renderer());
    // FIXME: [SVG] checkEnclosure implementation is inconsistent
    // https://bugs.webkit.org/show_bug.cgi?id=262709
    return intersectsAllowingEmpty(rect, ctm.mapRect(svgElement->checkedRenderer()->repaintRectInLocalCoordinates(RepaintRectCalculation::Accurate)));
}

bool LegacyRenderSVGModelObject::checkEnclosure(RenderElement* renderer, const FloatRect& rect)
{
    if (!renderer || renderer->usedPointerEvents() == PointerEvents::None)
        return false;
    if (!isGraphicsElement(*renderer))
        return false;
    AffineTransform ctm;
    RefPtr svgElement = downcast<SVGElement>(renderer->element());
    getElementCTM(svgElement.get(), ctm);
    ASSERT(svgElement->renderer());
    // FIXME: [SVG] checkEnclosure implementation is inconsistent
    // https://bugs.webkit.org/show_bug.cgi?id=262709
    return rect.contains(ctm.mapRect(svgElement->checkedRenderer()->repaintRectInLocalCoordinates(RepaintRectCalculation::Accurate)));
}

Ref<SVGElement> LegacyRenderSVGModelObject::protectedElement() const
{
    return element();
}

} // namespace WebCore
