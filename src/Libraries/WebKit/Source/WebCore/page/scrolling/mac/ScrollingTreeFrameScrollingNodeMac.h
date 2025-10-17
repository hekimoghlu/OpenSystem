/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#include "ScrollbarThemeMac.h"
#include "ScrollingStateFrameScrollingNode.h"
#include "ScrollingTreeFrameScrollingNode.h"
#include "ScrollingTreeScrollingNodeDelegateMac.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS CALayer;

namespace WebCore {

class ScrollingTreeScrollingNodeDelegateMac;

class WEBCORE_EXPORT ScrollingTreeFrameScrollingNodeMac : public ScrollingTreeFrameScrollingNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingTreeFrameScrollingNodeMac, WEBCORE_EXPORT);
public:
    static Ref<ScrollingTreeFrameScrollingNode> create(ScrollingTree&, ScrollingNodeType, ScrollingNodeID);
    virtual ~ScrollingTreeFrameScrollingNodeMac();

    RetainPtr<CALayer> rootContentsLayer() const { return m_rootContentsLayer; }

protected:
    ScrollingTreeFrameScrollingNodeMac(ScrollingTree&, ScrollingNodeType, ScrollingNodeID);

    // ScrollingTreeNode member functions.
    bool commitStateBeforeChildren(const ScrollingStateNode&) override;
    bool commitStateAfterChildren(const ScrollingStateNode&) override;

    WheelEventHandlingResult handleWheelEvent(const PlatformWheelEvent&, EventTargeting) override;

    WEBCORE_EXPORT void repositionRelatedLayers() override;

    FloatPoint minimumScrollPosition() const override;
    FloatPoint maximumScrollPosition() const override;

    void updateMainFramePinAndRubberbandState();

    unsigned exposedUnfilledArea() const;

private:
    ScrollingTreeScrollingNodeDelegateMac& delegate() const;

    void willBeDestroyed() final;
    void willDoProgrammaticScroll(const FloatPoint&) final;

    void currentScrollPositionChanged(ScrollType, ScrollingLayerPositionAction) final;
    void repositionScrollingLayers() final WTF_REQUIRES_LOCK(scrollingTree()->treeLock());

    RetainPtr<CALayer> m_rootContentsLayer;
    RetainPtr<CALayer> m_counterScrollingLayer;
    RetainPtr<CALayer> m_insetClipLayer;
    RetainPtr<CALayer> m_contentShadowLayer;
    RetainPtr<CALayer> m_headerLayer;
    RetainPtr<CALayer> m_footerLayer;
    
    bool m_lastScrollHadUnfilledPixels { false };
    bool m_hadFirstUpdate { false };
};

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)
