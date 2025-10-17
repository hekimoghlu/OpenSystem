/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#pragma once

#include "FrameView.h"
#include "RemoteFrame.h"

#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RemoteFrame;

class RemoteFrameView final : public FrameView {
    WTF_MAKE_TZONE_ALLOCATED(RemoteFrameView);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteFrameView);
public:
    static Ref<RemoteFrameView> create(RemoteFrame& frame) { return adoptRef(*new RemoteFrameView(frame)); }

    Type viewType() const final { return Type::Remote; }
    void writeRenderTreeAsText(TextStream&, OptionSet<RenderAsTextFlag>) override;
    RemoteFrame& frame() const final { return m_frame; }

private:
    WEBCORE_EXPORT RemoteFrameView(RemoteFrame&);

    bool isRemoteFrameView() const final { return true; }
    bool isScrollableOrRubberbandable() final;
    bool hasScrollableOrRubberbandableAncestor() final;
    bool shouldPlaceVerticalScrollbarOnLeft() const final;
    void invalidateScrollbarRect(Scrollbar&, const IntRect&) final;
    IntRect windowClipRect() const final;
    void paintContents(GraphicsContext&, const IntRect& damageRect, SecurityOriginPaintPolicy, RegionContext*) final;
    void addedOrRemovedScrollbar() final;
    void delegatedScrollingModeDidChange() final;
    void updateScrollCorner() final;
    bool scrollContentsFastPath(const IntSize& scrollDelta, const IntRect& rectToScroll, const IntRect& clipRect) final;
    bool isVerticalDocument() const final;
    bool isFlippedDocument() const final;
    bool shouldDeferScrollUpdateAfterContentSizeChange() final;
    void scrollOffsetChangedViaPlatformWidgetImpl(const ScrollOffset&, const ScrollOffset&) final;
    void unobscuredContentSizeChanged() final;
    void didFinishProhibitingScrollingWhenChangingContentSize() final;
    void updateLayerPositionsAfterScrolling() final;
    void updateCompositingLayersAfterScrolling() final;

    void setFrameRect(const IntRect&) final;

    const Ref<RemoteFrame> m_frame;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RemoteFrameView)
static bool isType(const WebCore::FrameView& view) { return view.viewType() == WebCore::FrameView::Type::Remote; }
static bool isType(const WebCore::Widget& widget) { return widget.isRemoteFrameView(); }
SPECIALIZE_TYPE_TRAITS_END()
