/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#include "RenderSVGContainer.h"

#include "GraphicsContext.h"
#include "HitTestRequest.h"
#include "HitTestResult.h"
#include "LayoutRepainter.h"
#include "RenderIterator.h"
#include "RenderLayer.h"
#include "RenderTreeBuilder.h"
#include "RenderView.h"
#include "SVGContainerLayout.h"
#include "SVGLayerTransformUpdater.h"
#include "SVGVisitedRendererTracking.h"
#include <wtf/SetForScope.h>
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGContainer);

RenderSVGContainer::RenderSVGContainer(Type type, Document& document, RenderStyle&& style, OptionSet<SVGModelObjectFlag> svgFlags)
    : RenderSVGModelObject(type, document, WTFMove(style), svgFlags | SVGModelObjectFlag::IsContainer)
{
}

RenderSVGContainer::RenderSVGContainer(Type type, SVGElement& element, RenderStyle&& style, OptionSet<SVGModelObjectFlag> svgFlags)
    : RenderSVGModelObject(type, element, WTFMove(style), svgFlags | SVGModelObjectFlag::IsContainer)
{
}

RenderSVGContainer::~RenderSVGContainer() = default;

void RenderSVGContainer::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());

    auto checkForRepaintOverride = isRenderSVGResourceMarker() ? std::make_optional(LayoutRepainter::CheckForRepaint::No) : std::nullopt;
    LayoutRepainter repainter(*this, checkForRepaintOverride);

    // Update layer transform before laying out children (SVG needs access to the transform matrices during layout for on-screen text font-size calculations).
    // Eventually re-update if the transform reference box, relevant for transform-origin, has changed during layout.
    //
    // FIXME: LBSE should not repeat the same mistake -- remove the on-screen text font-size hacks that predate the modern solutions to this.
    {
        ASSERT(!m_isLayoutSizeChanged);
        SetForScope trackLayoutSizeChanges(m_isLayoutSizeChanged, updateLayoutSizeIfNeeded());

        ASSERT(!m_didTransformToRootUpdate);
        SVGLayerTransformUpdater transformUpdater(*this);
        SetForScope trackTransformChanges(m_didTransformToRootUpdate, transformUpdater.layerTransformChanged() || SVGContainerLayout::transformToRootChanged(parent()));
        layoutChildren();
    }

    repainter.repaintAfterLayout();
    clearNeedsLayout();
}

void RenderSVGContainer::layoutChildren()
{
    SVGContainerLayout containerLayout(*this);
    containerLayout.layoutChildren(selfNeedsLayout());

    SVGBoundingBoxComputation boundingBoxComputation(*this);
    m_objectBoundingBox = boundingBoxComputation.computeDecoratedBoundingBox(SVGBoundingBoxComputation::objectBoundingBoxDecoration, &m_objectBoundingBoxValid);
    m_strokeBoundingBox = std::nullopt;

    if (auto objectBoundingBoxWithoutTransformations = overridenObjectBoundingBoxWithoutTransformations())
        m_objectBoundingBoxWithoutTransformations = objectBoundingBoxWithoutTransformations.value();
    else {
        constexpr auto objectBoundingBoxDecorationWithoutTransformations = SVGBoundingBoxComputation::objectBoundingBoxDecoration | SVGBoundingBoxComputation::DecorationOption::IgnoreTransformations;
        m_objectBoundingBoxWithoutTransformations = boundingBoxComputation.computeDecoratedBoundingBox(objectBoundingBoxDecorationWithoutTransformations);
    }

    setCurrentSVGLayoutRect(enclosingLayoutRect(m_objectBoundingBoxWithoutTransformations));

    containerLayout.positionChildrenRelativeToContainer();
}

FloatRect RenderSVGContainer::strokeBoundingBox() const
{
    if (!m_strokeBoundingBox) {
        // Initialize m_strokeBoundingBox before calling computeDecoratedBoundingBox, since recursively referenced markers can cause us to re-enter here.
        m_strokeBoundingBox = FloatRect { };
        SVGBoundingBoxComputation boundingBoxComputation(*this);
        m_strokeBoundingBox = boundingBoxComputation.computeDecoratedBoundingBox(SVGBoundingBoxComputation::strokeBoundingBoxDecoration);
    }
    return *m_strokeBoundingBox;
}

void RenderSVGContainer::paint(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    OptionSet<PaintPhase> relevantPaintPhases { PaintPhase::Foreground, PaintPhase::ClippingMask, PaintPhase::Mask, PaintPhase::Outline, PaintPhase::SelfOutline };
    if (!shouldPaintSVGRenderer(paintInfo, relevantPaintPhases))
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
}

bool RenderSVGContainer::nodeAtPoint(const HitTestRequest& request, HitTestResult& result, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction hitTestAction)
{
    auto adjustedLocation = accumulatedOffset + currentSVGLayoutLocation();

    auto visualOverflowRect = visualOverflowRectEquivalent();
    visualOverflowRect.moveBy(adjustedLocation);
    if (!locationInContainer.intersects(visualOverflowRect))
        return false;

    static NeverDestroyed<SVGVisitedRendererTracking::VisitedSet> s_visitedSet;

    SVGVisitedRendererTracking recursionTracking(s_visitedSet);
    if (recursionTracking.isVisiting(*this))
        return false;

    SVGVisitedRendererTracking::Scope recursionScope(recursionTracking, *this);

    auto localPoint = locationInContainer.point();
    auto coordinateSystemOriginTranslation = nominalSVGLayoutLocation() - adjustedLocation;
    localPoint.move(coordinateSystemOriginTranslation);

    if (!pointInSVGClippingArea(localPoint))
        return false;

    // Give RenderSVGViewportContainer a chance to apply its viewport clip.
    if (!pointIsInsideViewportClip(localPoint))
        return false;


    for (auto* child = lastChild(); child; child = child->previousSibling()) {
        if (!child->hasLayer() && child->nodeAtPoint(request, result, locationInContainer, adjustedLocation, hitTestAction)) {
            updateHitTestResult(result, locationInContainer.point() - toLayoutSize(adjustedLocation));
            if (result.addNodeToListBasedTestResult(child->protectedNode().get(), request, locationInContainer, visualOverflowRect) == HitTestProgress::Stop)
                return true;
        }
    }

    // Accessibility wants to return SVG containers, if appropriate.
    if (request.type() & HitTestRequest::Type::AccessibilityHitTest && m_objectBoundingBox.contains(localPoint)) {
        updateHitTestResult(result, locationInContainer.point() - toLayoutSize(adjustedLocation));
        if (result.addNodeToListBasedTestResult(protectedNodeForHitTest().get(), request, locationInContainer, visualOverflowRect) == HitTestProgress::Stop)
            return true;
    }

    // Spec: Only graphical elements can be targeted by the mouse, period.
    // 16.4: "If there are no graphics elements whose relevant graphics content is under the pointer (i.e., there is no target element), the event is not dispatched."
    return false;
}

}

