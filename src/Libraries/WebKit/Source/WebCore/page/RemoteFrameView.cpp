/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#include "RemoteFrameView.h"

#include "RemoteFrame.h"
#include "RemoteFrameClient.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteFrameView);

RemoteFrameView::RemoteFrameView(RemoteFrame& frame)
    : m_frame(frame)
{
}

void RemoteFrameView::setFrameRect(const IntRect& newRect)
{
    IntRect oldRect = frameRect();
    if (newRect.size() != oldRect.size())
        m_frame->client().sizeDidChange(newRect.size());
    FrameView::setFrameRect(newRect);
}

// FIXME: Implement all the stubs below.

bool RemoteFrameView::isScrollableOrRubberbandable()
{
    return false;
}

bool RemoteFrameView::hasScrollableOrRubberbandableAncestor()
{
    return false;
}

bool RemoteFrameView::shouldPlaceVerticalScrollbarOnLeft() const
{
    return false;
}

void RemoteFrameView::invalidateScrollbarRect(Scrollbar&, const IntRect&)
{
}

IntRect RemoteFrameView::windowClipRect() const
{
    return { };
}

void RemoteFrameView::paintContents(GraphicsContext&, const IntRect&, SecurityOriginPaintPolicy, RegionContext*)
{
}

void RemoteFrameView::addedOrRemovedScrollbar()
{
}

void RemoteFrameView::delegatedScrollingModeDidChange()
{
}

void RemoteFrameView::updateScrollCorner()
{
}

bool RemoteFrameView::scrollContentsFastPath(const IntSize&, const IntRect&, const IntRect&)
{
    return false;
}

bool RemoteFrameView::isVerticalDocument() const
{
    return false;
}

bool RemoteFrameView::isFlippedDocument() const
{
    return false;
}

bool RemoteFrameView::shouldDeferScrollUpdateAfterContentSizeChange()
{
    return false;
}

void RemoteFrameView::scrollOffsetChangedViaPlatformWidgetImpl(const ScrollOffset&, const ScrollOffset&)
{
}

void RemoteFrameView::unobscuredContentSizeChanged()
{
}

void RemoteFrameView::didFinishProhibitingScrollingWhenChangingContentSize()
{
}

void RemoteFrameView::updateLayerPositionsAfterScrolling()
{
}

void RemoteFrameView::updateCompositingLayersAfterScrolling()
{
}

void RemoteFrameView::writeRenderTreeAsText(TextStream& ts, OptionSet<RenderAsTextFlag> behavior)
{
    auto& remoteFrame = frame();
    ts << remoteFrame.renderTreeAsText(ts.indent(), behavior);
}

} // namespace WebCore
