/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#include "LegacyRenderSVGForeignObject.h"

#include "GraphicsContext.h"
#include "HitTestResult.h"
#include "LayoutRepainter.h"
#include "LegacyRenderSVGResource.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderSVGBlockInlines.h"
#include "RenderView.h"
#include "SVGElementTypeHelpers.h"
#include "SVGForeignObjectElement.h"
#include "SVGRenderSupport.h"
#include "SVGRenderingContext.h"
#include "SVGResourcesCache.h"
#include "TransformState.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGForeignObject);

LegacyRenderSVGForeignObject::LegacyRenderSVGForeignObject(SVGForeignObjectElement& element, RenderStyle&& style)
    : RenderSVGBlock(Type::LegacySVGForeignObject, element, WTFMove(style))
{
    ASSERT(isLegacyRenderSVGForeignObject());
}

LegacyRenderSVGForeignObject::~LegacyRenderSVGForeignObject() = default;

SVGForeignObjectElement& LegacyRenderSVGForeignObject::foreignObjectElement() const
{
    return downcast<SVGForeignObjectElement>(RenderSVGBlock::graphicsElement());
}

void LegacyRenderSVGForeignObject::paint(PaintInfo& paintInfo, const LayoutPoint&)
{
    if (paintInfo.context().paintingDisabled())
        return;

    if (paintInfo.phase != PaintPhase::Foreground && paintInfo.phase != PaintPhase::Selection)
        return;

    PaintInfo childPaintInfo(paintInfo);
    GraphicsContextStateSaver stateSaver(childPaintInfo.context());
    childPaintInfo.applyTransform(localTransform());

    if (SVGRenderSupport::isOverflowHidden(*this))
        childPaintInfo.context().clip(m_viewport);

    SVGRenderingContext renderingContext;
    if (paintInfo.phase == PaintPhase::Foreground) {
        renderingContext.prepareToRenderSVGContent(*this, childPaintInfo);
        if (!renderingContext.isRenderingPrepared())
            return;
    }

    LayoutPoint childPoint = IntPoint();
    if (paintInfo.phase == PaintPhase::Selection) {
        RenderBlock::paint(childPaintInfo, childPoint);
        return;
    }

    // Paint all phases of FO elements atomically, as though the FO element established its
    // own stacking context.
    childPaintInfo.phase = PaintPhase::BlockBackground;
    RenderBlock::paint(childPaintInfo, childPoint);
    childPaintInfo.phase = PaintPhase::ChildBlockBackgrounds;
    RenderBlock::paint(childPaintInfo, childPoint);
    childPaintInfo.phase = PaintPhase::Float;
    RenderBlock::paint(childPaintInfo, childPoint);
    childPaintInfo.phase = PaintPhase::Foreground;
    RenderBlock::paint(childPaintInfo, childPoint);
    childPaintInfo.phase = PaintPhase::Outline;
    RenderBlock::paint(childPaintInfo, childPoint);
}

const AffineTransform& LegacyRenderSVGForeignObject::localToParentTransform() const
{
    m_localToParentTransform = localTransform();
    m_localToParentTransform.translate(m_viewport.location());
    return m_localToParentTransform;
}

void LegacyRenderSVGForeignObject::updateLogicalWidth()
{
    // FIXME: Investigate in size rounding issues
    // FIXME: Remove unnecessary rounding when layout is off ints: webkit.org/b/63656
    setWidth(static_cast<int>(roundf(m_viewport.width())));
}

RenderBox::LogicalExtentComputedValues LegacyRenderSVGForeignObject::computeLogicalHeight(LayoutUnit, LayoutUnit logicalTop) const
{
    // FIXME: Investigate in size rounding issues
    // FIXME: Remove unnecessary rounding when layout is off ints: webkit.org/b/63656
    // FIXME: Is this correct for vertical writing mode?
    return { static_cast<int>(roundf(m_viewport.height())), logicalTop, ComputedMarginValues() };
}

void LegacyRenderSVGForeignObject::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());
    ASSERT(!view().frameView().layoutContext().isPaintOffsetCacheEnabled()); // LegacyRenderSVGRoot disables paint offset cache for the SVG rendering tree.

    LayoutRepainter repainter(*this, SVGRenderSupport::checkForSVGRepaintDuringLayout(*this));

    bool updateCachedBoundariesInParents = false;
    if (m_needsTransformUpdate) {
        m_localTransform = foreignObjectElement().animatedLocalTransform();
        m_needsTransformUpdate = false;
        updateCachedBoundariesInParents = true;
    }

    FloatRect oldViewport = m_viewport;

    // Cache viewport boundaries
    Ref foreignObjectElement = this->foreignObjectElement();
    SVGLengthContext lengthContext(foreignObjectElement.ptr());
    FloatPoint viewportLocation(foreignObjectElement->x().value(lengthContext), foreignObjectElement->y().value(lengthContext));
    m_viewport = FloatRect(viewportLocation, FloatSize(foreignObjectElement->width().value(lengthContext), foreignObjectElement->height().value(lengthContext)));
    if (!updateCachedBoundariesInParents)
        updateCachedBoundariesInParents = oldViewport != m_viewport;

    // Set box origin to the foreignObject x/y translation, so positioned objects in XHTML content get correct
    // positions. A regular RenderBoxModelObject would pull this information from RenderStyle - in SVG those
    // properties are ignored for non <svg> elements, so we mimic what happens when specifying them through CSS.

    // FIXME: Investigate in location rounding issues - only affects LegacyRenderSVGForeignObject & RenderSVGText
    setLocation(roundedIntPoint(viewportLocation));

    bool layoutChanged = everHadLayout() && selfNeedsLayout();
    RenderBlock::layout();
    ASSERT(!needsLayout());

    // If our bounds changed, notify the parents.
    if (updateCachedBoundariesInParents) {
        if (CheckedPtr parent = this->parent())
            parent->invalidateCachedBoundaries();
    }

    // Invalidate all resources of this client if our layout changed.
    if (layoutChanged)
        SVGResourcesCache::clientLayoutChanged(*this);

    repainter.repaintAfterLayout();
}

bool LegacyRenderSVGForeignObject::nodeAtFloatPoint(const HitTestRequest& request, HitTestResult& result, const FloatPoint& pointInParent, HitTestAction hitTestAction)
{
    // Embedded content is drawn in the foreground phase.
    if (hitTestAction != HitTestForeground)
        return false;

    FloatPoint localPoint = valueOrDefault(localTransform().inverse()).mapPoint(pointInParent);

    // Early exit if local point is not contained in clipped viewport area
    if (SVGRenderSupport::isOverflowHidden(*this) && !m_viewport.contains(localPoint))
        return false;

    // FOs establish a stacking context, so we need to hit-test all layers.
    HitTestLocation hitTestLocation(flooredLayoutPoint(localPoint));
    return RenderBlock::nodeAtPoint(request, result, hitTestLocation, LayoutPoint(), HitTestForeground)
        || RenderBlock::nodeAtPoint(request, result, hitTestLocation, LayoutPoint(), HitTestFloat)
        || RenderBlock::nodeAtPoint(request, result, hitTestLocation, LayoutPoint(), HitTestChildBlockBackgrounds);
}

LayoutSize LegacyRenderSVGForeignObject::offsetFromContainer(RenderElement& container, const LayoutPoint&, bool*) const
{
    ASSERT_UNUSED(container, &container == this->container());
    ASSERT(!isInFlowPositioned());
    ASSERT(!isAbsolutelyPositioned());
    ASSERT(!isInline());
    return locationOffset();
}

}
