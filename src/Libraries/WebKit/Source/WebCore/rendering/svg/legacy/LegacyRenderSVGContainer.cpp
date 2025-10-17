/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#include "LegacyRenderSVGContainer.h"

#include "GraphicsContext.h"
#include "HitTestRequest.h"
#include "HitTestResult.h"
#include "LayoutRepainter.h"
#include "RenderIterator.h"
#include "RenderTreeBuilder.h"
#include "RenderView.h"
#include "SVGRenderingContext.h"
#include "SVGResources.h"
#include "SVGResourcesCache.h"
#include "SVGVisitedRendererTracking.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGContainer);

static bool shouldSuspendRepaintForChildren(const LegacyRenderSVGContainer& container)
{
    // Issuing repaint requires absolute rect which means ancestor tree walk.
    // In cases when a container has many direct children, the overhead of resolving each individual repaint rects
    // is so large that issuing repaint for the container itself ends up being cheaper.
    static constexpr size_t maximumRequiredChildren = 200;
    static constexpr size_t minimumRequiredChildren = 50;

    size_t numberOfChildren = 0;
    for (auto& child : childrenOfType<RenderElement>(container)) {
        ++numberOfChildren;
        if (!child.needsLayout() || child.firstChild())
            return false;
        if (numberOfChildren >= maximumRequiredChildren)
            return true;
    }
    return numberOfChildren >= minimumRequiredChildren;
}

LegacyRenderSVGContainer::LegacyRenderSVGContainer(Type type, SVGElement& element, RenderStyle&& style, OptionSet<SVGModelObjectFlag> svgFlags)
    : LegacyRenderSVGModelObject(type, element, WTFMove(style), svgFlags | SVGModelObjectFlag::IsContainer | SVGModelObjectFlag::UsesBoundaryCaching)
{
}

LegacyRenderSVGContainer::~LegacyRenderSVGContainer() = default;

void LegacyRenderSVGContainer::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());

    // LegacyRenderSVGRoot disables paint offset cache for the SVG rendering tree.
    ASSERT(!view().frameView().layoutContext().isPaintOffsetCacheEnabled());
    auto repaintIsSuspendedForChildrenDuringLayoutScope = SetForScope { m_repaintIsSuspendedForChildrenDuringLayout, shouldSuspendRepaintForChildren(*this) };

    auto checkForRepaintOverride = m_repaintIsSuspendedForChildrenDuringLayout || selfWillPaint() ? LayoutRepainter::CheckForRepaint::Yes : SVGRenderSupport::checkForSVGRepaintDuringLayout(*this);
    auto shouldIssueFullRepaint = m_repaintIsSuspendedForChildrenDuringLayout ? LayoutRepainter::ShouldAlwaysIssueFullRepaint::Yes : LayoutRepainter::ShouldAlwaysIssueFullRepaint::No;
    LayoutRepainter repainter(*this, checkForRepaintOverride, shouldIssueFullRepaint, RepaintOutlineBounds::No);

    // Allow LegacyRenderSVGViewportContainer to update its viewport.
    calcViewport();

    // Allow LegacyRenderSVGTransformableContainer to update its transform.
    bool updatedTransform = calculateLocalTransform();

    // LegacyRenderSVGViewportContainer needs to set the 'layout size changed' flag.
    determineIfLayoutSizeChanged();

    SVGRenderSupport::layoutChildren(*this, selfNeedsLayout() || SVGRenderSupport::filtersForceContainerLayout(*this));

    // Invalidate all resources of this client if our layout changed.
    if (everHadLayout() && needsLayout())
        SVGResourcesCache::clientLayoutChanged(*this);

    // At this point LayoutRepainter already grabbed the old bounds,
    // recalculate them now so repaintAfterLayout() uses the new bounds.
    if (m_needsBoundariesUpdate || updatedTransform) {
        updateCachedBoundaries();
        m_needsBoundariesUpdate = false;

        // New bounds can affect transforms, so recompute them here if needed.
        calculateLocalTransform();

        if (CheckedPtr parent = this->parent())
            parent->invalidateCachedBoundaries();
    }

    repainter.repaintAfterLayout();
    clearNeedsLayout();
}

bool LegacyRenderSVGContainer::selfWillPaint()
{
    auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*this);
    return resources && resources->filter();
}

void LegacyRenderSVGContainer::paint(PaintInfo& paintInfo, const LayoutPoint&)
{
    if (paintInfo.phase != PaintPhase::EventRegion && paintInfo.context().paintingDisabled())
        return;

    // Spec: groups w/o children still may render filter content.
    if (!firstChild() && !selfWillPaint())
        return;

    FloatRect repaintRect = repaintRectInLocalCoordinates();
    if (!SVGRenderSupport::paintInfoIntersectsRepaintRect(repaintRect, localToParentTransform(), paintInfo))
        return;

    PaintInfo childPaintInfo(paintInfo);
    {
        GraphicsContextStateSaver stateSaver(childPaintInfo.context());

        // Let the LegacyRenderSVGViewportContainer subclass clip if necessary
        applyViewportClip(childPaintInfo);

        auto transform = localToParentTransform();
        childPaintInfo.applyTransform(transform);
        if (paintInfo.phase == PaintPhase::EventRegion && childPaintInfo.eventRegionContext())
            childPaintInfo.eventRegionContext()->pushTransform(transform);

        SVGRenderingContext renderingContext;
        bool continueRendering = true;
        if (childPaintInfo.phase == PaintPhase::Foreground) {
            renderingContext.prepareToRenderSVGContent(*this, childPaintInfo);
            continueRendering = renderingContext.isRenderingPrepared();
        }

        if (continueRendering) {
            childPaintInfo.updateSubtreePaintRootForChildren(this);
            for (auto& child : childrenOfType<RenderElement>(*this))
                child.paint(childPaintInfo, IntPoint());
        }

        if (paintInfo.phase == PaintPhase::EventRegion && childPaintInfo.eventRegionContext())
            childPaintInfo.eventRegionContext()->popTransform();
    }

    
    // FIXME: This really should be drawn from local coordinates, but currently we hack it
    // to avoid our clip killing our outline rect. Thus we translate our
    // outline rect into parent coords before drawing.
    // FIXME: This means our focus ring won't share our rotation like it should.
    // We should instead disable our clip during PaintPhase::Outline
    if (paintInfo.phase == PaintPhase::SelfOutline && style().outlineWidth() && style().usedVisibility() == Visibility::Visible) {
        IntRect paintRectInParent = enclosingIntRect(localToParentTransform().mapRect(repaintRect));
        paintOutline(paintInfo, paintRectInParent);
    }
}

// addFocusRingRects is called from paintOutline and needs to be in the same coordinates as the paintOuline call
void LegacyRenderSVGContainer::addFocusRingRects(Vector<LayoutRect>& rects, const LayoutPoint&, const RenderLayerModelObject*) const
{
    LayoutRect paintRectInParent = LayoutRect(localToParentTransform().mapRect(repaintRectInLocalCoordinates()));
    if (!paintRectInParent.isEmpty())
        rects.append(paintRectInParent);
}

void LegacyRenderSVGContainer::updateCachedBoundaries()
{
    m_strokeBoundingBox = std::nullopt;
    m_repaintBoundingBox = { };
    m_accurateRepaintBoundingBox = std::nullopt;
    FloatRect repaintBoundingBox;
    SVGRenderSupport::computeContainerBoundingBoxes(*this, m_objectBoundingBox, m_objectBoundingBoxValid, repaintBoundingBox);
    SVGRenderSupport::intersectRepaintRectWithResources(*this, repaintBoundingBox);
    m_repaintBoundingBox = repaintBoundingBox;
}

FloatRect LegacyRenderSVGContainer::strokeBoundingBox() const
{
    if (!m_strokeBoundingBox) {
        // Initialize m_strokeBoundingBox before calling computeContainerStrokeBoundingBox, since recursively referenced markers can cause us to re-enter here.
        m_strokeBoundingBox = FloatRect { };
        m_strokeBoundingBox = SVGRenderSupport::computeContainerStrokeBoundingBox(*this);
    }
    return *m_strokeBoundingBox;
}

FloatRect LegacyRenderSVGContainer::repaintRectInLocalCoordinates(RepaintRectCalculation repaintRectCalculation) const
{
    if (repaintRectCalculation == RepaintRectCalculation::Fast)
        return m_repaintBoundingBox;

    if (!m_accurateRepaintBoundingBox) {
        // Initialize m_accurateRepaintBoundingBox before calling computeContainerBoundingBoxes, since recursively referenced markers can cause us to re-enter here.
        m_accurateRepaintBoundingBox = FloatRect { };
        FloatRect objectBoundingBox;
        FloatRect repaintBoundingBox;
        bool objectBoundingBoxValid = true;
        SVGRenderSupport::computeContainerBoundingBoxes(*this, objectBoundingBox, objectBoundingBoxValid, repaintBoundingBox, RepaintRectCalculation::Accurate);
        SVGRenderSupport::intersectRepaintRectWithResources(*this, repaintBoundingBox, RepaintRectCalculation::Accurate);
        m_accurateRepaintBoundingBox = repaintBoundingBox;
    }
    return *m_accurateRepaintBoundingBox;
}

bool LegacyRenderSVGContainer::nodeAtFloatPoint(const HitTestRequest& request, HitTestResult& result, const FloatPoint& pointInParent, HitTestAction hitTestAction)
{
    // Give LegacyRenderSVGViewportContainer a chance to apply its viewport clip
    if (!pointIsInsideViewportClip(pointInParent))
        return false;

    static NeverDestroyed<SVGVisitedRendererTracking::VisitedSet> s_visitedSet;

    SVGVisitedRendererTracking recursionTracking(s_visitedSet);
    if (recursionTracking.isVisiting(*this))
        return false;

    SVGVisitedRendererTracking::Scope recursionScope(recursionTracking, *this);

    FloatPoint localPoint = valueOrDefault(localToParentTransform().inverse()).mapPoint(pointInParent);
    if (!SVGRenderSupport::pointInClippingArea(*this, localPoint))
        return false;

    for (RenderObject* child = lastChild(); child; child = child->previousSibling()) {
        if (child->nodeAtFloatPoint(request, result, localPoint, hitTestAction)) {
            updateHitTestResult(result, LayoutPoint(localPoint));
            if (result.addNodeToListBasedTestResult(child->protectedNode().get(), request, flooredLayoutPoint(localPoint)) == HitTestProgress::Stop)
                return true;
        }
    }

    // Accessibility wants to return SVG containers, if appropriate.
    if (request.type() & HitTestRequest::Type::AccessibilityHitTest && m_objectBoundingBox.contains(localPoint)) {
        updateHitTestResult(result, LayoutPoint(localPoint));
        if (result.addNodeToListBasedTestResult(protectedNodeForHitTest().get(), request, flooredLayoutPoint(localPoint)) == HitTestProgress::Stop)
            return true;
    }
    
    // Spec: Only graphical elements can be targeted by the mouse, period.
    // 16.4: "If there are no graphics elements whose relevant graphics content is under the pointer (i.e., there is no target element), the event is not dispatched."
    return false;
}

}
