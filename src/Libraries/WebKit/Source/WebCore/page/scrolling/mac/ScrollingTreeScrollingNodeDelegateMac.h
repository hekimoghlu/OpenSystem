/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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

#include "ScrollerPairMac.h"
#include "ScrollingEffectsController.h"
#include "ThreadedScrollingTreeScrollingNodeDelegate.h"
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS NSScrollerImp;

namespace WebCore {

class FloatPoint;
class FloatSize;
class IntPoint;
class ScrollingStateScrollingNode;
class ScrollingTreeScrollingNode;
class ScrollingTree;

class ScrollingTreeScrollingNodeDelegateMac final : public ThreadedScrollingTreeScrollingNodeDelegate {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreeScrollingNodeDelegateMac);
public:
    explicit ScrollingTreeScrollingNodeDelegateMac(ScrollingTreeScrollingNode&);
    virtual ~ScrollingTreeScrollingNodeDelegateMac();

    void nodeWillBeDestroyed();

    bool handleWheelEvent(const PlatformWheelEvent&);

    void willDoProgrammaticScroll(const FloatPoint&);
    void currentScrollPositionChanged();

    bool isRubberBandInProgress() const;

    void updateScrollbarPainters();
    void updateScrollbarLayers() final;
    
    void handleWheelEventPhase(const PlatformWheelEventPhase) final;
    void viewWillStartLiveResize() final;
    void viewWillEndLiveResize() final;
    void viewSizeDidChange() final;
    void initScrollbars() final;
    String scrollbarStateForOrientation(ScrollbarOrientation) const final;

private:
    void updateFromStateNode(const ScrollingStateScrollingNode&) final;

    // ScrollingEffectsControllerClient.
    bool allowsHorizontalStretching(const PlatformWheelEvent&) const final;
    bool allowsVerticalStretching(const PlatformWheelEvent&) const final;
    IntSize stretchAmount() const final;
    bool isPinnedOnSide(BoxSide) const final;
    RectEdges<bool> edgePinnedState() const final;

    bool shouldRubberBandOnSide(BoxSide) const final;
    void didStopRubberBandAnimation() final;
    void rubberBandingStateChanged(bool) final;
    bool scrollPositionIsNotRubberbandingEdge(const FloatPoint&) const;

    Ref<ScrollerPairMac> m_scrollerPair;

    bool m_inMomentumPhase { false };
};

} // namespace WebCore

#endif // PLATFORM(MAC) && ENABLE(ASYNC_SCROLLING)
