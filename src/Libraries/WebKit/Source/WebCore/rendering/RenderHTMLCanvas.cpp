/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include "RenderHTMLCanvas.h"

#include "CanvasRenderingContext.h"
#include "Document.h"
#include "GraphicsContext.h"
#include "HTMLCanvasElement.h"
#include "HTMLNames.h"
#include "ImageQualityController.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "PaintInfo.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include "RenderLayerBacking.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderHTMLCanvas);

RenderHTMLCanvas::RenderHTMLCanvas(HTMLCanvasElement& element, RenderStyle&& style)
    : RenderReplaced(Type::HTMLCanvas, element, WTFMove(style), element.size())
{
    ASSERT(isRenderHTMLCanvas());
}

RenderHTMLCanvas::~RenderHTMLCanvas() = default;

HTMLCanvasElement& RenderHTMLCanvas::canvasElement() const
{
    return downcast<HTMLCanvasElement>(nodeForNonAnonymous());
}

bool RenderHTMLCanvas::requiresLayer() const
{
    if (RenderReplaced::requiresLayer())
        return true;

    return canvasCompositingStrategy(*this) != CanvasPaintedToEnclosingLayer;
}

void RenderHTMLCanvas::paintReplaced(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    GraphicsContext& context = paintInfo.context();

    LayoutRect contentBoxRect = this->contentBoxRect();

    if (context.detectingContentfulPaint()) {
        if (!context.contentfulPaintDetected() && canvasElement().renderingContext())
            context.setContentfulPaintDetected();
        return;
    }

    contentBoxRect.moveBy(paintOffset);
    LayoutRect replacedContentRect = this->replacedContentRect();
    replacedContentRect.moveBy(paintOffset);

    // Not allowed to overflow the content box.
    bool clip = !contentBoxRect.contains(replacedContentRect);
    GraphicsContextStateSaver stateSaver(paintInfo.context(), clip);
    if (clip)
        paintInfo.context().clip(snappedIntRect(contentBoxRect));

    if (paintInfo.phase == PaintPhase::Foreground)
        page().addRelevantRepaintedObject(*this, intersection(replacedContentRect, contentBoxRect));

    InterpolationQualityMaintainer interpolationMaintainer(context, ImageQualityController::interpolationQualityFromStyle(style()));

    canvasElement().setIsSnapshotting(paintInfo.paintBehavior.contains(PaintBehavior::Snapshotting));
    canvasElement().paint(context, replacedContentRect);
    canvasElement().setIsSnapshotting(false);
}

void RenderHTMLCanvas::canvasSizeChanged()
{
    IntSize canvasSize = canvasElement().size();
    LayoutSize zoomedSize(canvasSize.width() * style().usedZoom(), canvasSize.height() * style().usedZoom());

    if (zoomedSize == intrinsicSize())
        return;

    setIntrinsicSize(zoomedSize);

    if (!parent())
        return;
    setNeedsLayoutIfNeededAfterIntrinsicSizeChange();
}

} // namespace WebCore
