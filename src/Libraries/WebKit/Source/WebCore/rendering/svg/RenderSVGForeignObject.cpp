/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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
#include "RenderSVGForeignObject.h"

#include "GraphicsContext.h"
#include "HitTestResult.h"
#include "LayoutRepainter.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include "RenderObject.h"
#include "RenderSVGBlockInlines.h"
#include "RenderView.h"
#include "SVGElementTypeHelpers.h"
#include "SVGForeignObjectElement.h"
#include "SVGRenderSupport.h"
#include "TransformState.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGForeignObject);

RenderSVGForeignObject::RenderSVGForeignObject(SVGForeignObjectElement& element, RenderStyle&& style)
    : RenderSVGBlock(Type::SVGForeignObject, element, WTFMove(style))
{
    ASSERT(isRenderSVGForeignObject());
}

RenderSVGForeignObject::~RenderSVGForeignObject() = default;

SVGForeignObjectElement& RenderSVGForeignObject::foreignObjectElement() const
{
    return downcast<SVGForeignObjectElement>(RenderSVGBlock::graphicsElement());
}

Ref<SVGForeignObjectElement> RenderSVGForeignObject::protectedForeignObjectElement() const
{
    return foreignObjectElement();
}

void RenderSVGForeignObject::paint(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    if (!shouldPaintSVGRenderer(paintInfo))
        return;

    if (paintInfo.phase == PaintPhase::ClippingMask) {
        paintSVGClippingMask(paintInfo, objectBoundingBox());
        return;
    }

    auto adjustedPaintOffset = paintOffset + location();
    if (paintInfo.phase == PaintPhase::Mask) {
        paintSVGMask(paintInfo, adjustedPaintOffset);
        return;
    }

    GraphicsContextStateSaver stateSaver(paintInfo.context());
    paintInfo.context().translate(adjustedPaintOffset.x(), adjustedPaintOffset.y());
    RenderSVGBlock::paint(paintInfo, paintOffset);
}

void RenderSVGForeignObject::updateLogicalWidth()
{
    setWidth(enclosingLayoutRect(m_viewport).width());
}

RenderBox::LogicalExtentComputedValues RenderSVGForeignObject::computeLogicalHeight(LayoutUnit, LayoutUnit logicalTop) const
{
    return { enclosingLayoutRect(m_viewport).height(), logicalTop, ComputedMarginValues() };
}

void RenderSVGForeignObject::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());

    LayoutRepainter repainter(*this);

    Ref useForeignObjectElement = foreignObjectElement();
    SVGLengthContext lengthContext(useForeignObjectElement.ptr());

    // Cache viewport boundaries
    auto x = useForeignObjectElement->x().value(lengthContext);
    auto y = useForeignObjectElement->y().value(lengthContext);
    auto width = useForeignObjectElement->width().value(lengthContext);
    auto height = useForeignObjectElement->height().value(lengthContext);
    m_viewport = { x, y, width, height };

    RenderSVGBlock::layout();
    ASSERT(!needsLayout());

    setLocation(enclosingLayoutRect(m_viewport).location());
    updateLayerTransform();

    repainter.repaintAfterLayout();
}

LayoutRect RenderSVGForeignObject::overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy, PaintPhase) const
{
    return enclosingLayoutRect(LayoutRect { location, m_viewport.size() });
}

void RenderSVGForeignObject::updateFromStyle()
{
    RenderSVGBlock::updateFromStyle();

    if (SVGRenderSupport::isOverflowHidden(*this))
        setHasNonVisibleOverflow();
}

void RenderSVGForeignObject::applyTransform(TransformationMatrix& transform, const RenderStyle& style, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption> options) const
{
    applySVGTransform(transform, protectedForeignObjectElement(), style, boundingBox, std::nullopt, std::nullopt, options);
}

}
