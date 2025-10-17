/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#include "AccessibilityRegionContext.h"

#include "AXObjectCache.h"
#include "DocumentInlines.h"
#include "RenderBox.h"
#include "RenderBoxModelObject.h"
#include "RenderInline.h"
#include "RenderStyleInlines.h"
#include "RenderText.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AccessibilityRegionContext);

AccessibilityRegionContext::~AccessibilityRegionContext()
{
    // Release all accumulated RenderText frames to the cache.
    for (auto renderTextToFrame : m_accumulatedRenderTextRects) {
        if (auto* cache = renderTextToFrame.key.document().axObjectCache())
            cache->onPaint(renderTextToFrame.key, enclosingIntRect(renderTextToFrame.value));
    }
}

void AccessibilityRegionContext::takeBounds(const RenderInline& renderInline, LayoutRect&& paintRect)
{
    takeBoundsInternal(renderInline, enclosingIntRect(mapRect(WTFMove(paintRect))));
}

void AccessibilityRegionContext::takeBounds(const RenderBox& renderBox, LayoutPoint paintOffset)
{
    if (CheckedPtr renderView = dynamicDowncast<RenderView>(renderBox); UNLIKELY(renderView)) {
        takeBounds(*renderView, WTFMove(paintOffset));
        return;
    }
    auto mappedPaintRect = enclosingIntRect(mapRect(LayoutRect(paintOffset, renderBox.size())));
    takeBoundsInternal(renderBox, WTFMove(mappedPaintRect));
}

void AccessibilityRegionContext::takeBounds(const RenderView& renderView, LayoutPoint&& paintOffset)
{
    // RenderView::documentRect applies necessary transforms. Intentionally do not apply the clips stored in this context since accessibility
    // expects the size to be that of the full document (not the size of the current viewport).
    auto documentRect = renderView.documentRect();
    documentRect.moveBy(roundedIntPoint(paintOffset));
    takeBoundsInternal(renderView, WTFMove(documentRect));
}

void AccessibilityRegionContext::takeBounds(const RenderBox& renderBox, FloatRect paintRect)
{
    auto mappedPaintRect = enclosingIntRect(mapRect(paintRect));
    takeBoundsInternal(renderBox, WTFMove(mappedPaintRect));
}

void AccessibilityRegionContext::takeBoundsInternal(const RenderBoxModelObject& renderObject, IntRect&& paintRect)
{
    if (auto* view = renderObject.document().view())
        paintRect = view->contentsToRootView(paintRect);

    if (auto* cache = renderObject.document().axObjectCache())
        cache->onPaint(renderObject, WTFMove(paintRect));
}

// Note that this function takes the bounds of a textbox that is associated with a RenderText, and not the RenderText itself.
// RenderTexts are not painted atomically. Instead, they are painted as multiple `InlineIterator::TextBox`s, where a textbox might represent
// the text on a single line. This method takes the paint rect of a single textbox and unites it with the other textbox rects painted for |renderText|.
void AccessibilityRegionContext::takeBounds(const RenderText& renderText, FloatRect paintRect)
{
    auto mappedPaintRect = enclosingIntRect(mapRect(WTFMove(paintRect)));
    if (auto* view = renderText.document().view())
        mappedPaintRect = view->contentsToRootView(mappedPaintRect);

    auto accumulatedRectIterator = m_accumulatedRenderTextRects.find(renderText);
    if (accumulatedRectIterator == m_accumulatedRenderTextRects.end())
        m_accumulatedRenderTextRects.set(renderText, mappedPaintRect);
    else
        accumulatedRectIterator->value.unite(mappedPaintRect);
}

void AccessibilityRegionContext::onPaint(const ScrollView& scrollView)
{
    auto* frameView = dynamicDowncast<LocalFrameView>(scrollView);
    if (!frameView) {
        // We require a frameview here because ScrollView has no existing mechanism to get an associated document,
        // and we need that to make the frame viewport-relative and to get an AXObjectCache.
        ASSERT_NOT_REACHED_WITH_MESSAGE("Tried to paint a scrollview that was not a frameview, so no frame will be computed");
        return;
    }

    auto relativeFrame = frameView->frameRectShrunkByInset();
    // Only normalize the rect to the root view if this scrollview isn't already associated with the root view (i.e. it has a frame owner).
    if (auto* frameOwnerElement = frameView->frame().ownerElement()) {
        if (auto* ownerDocumentFrameView = frameOwnerElement->document().view())
            relativeFrame = ownerDocumentFrameView->contentsToRootView(relativeFrame);
    }

    if (auto* cache = frameView->axObjectCache())
        cache->onPaint(*frameView, WTFMove(relativeFrame));
}

} // namespace WebCore
