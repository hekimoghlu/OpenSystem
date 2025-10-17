/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#import "config.h"

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)

#import "ScrollingCoordinatorMac.h"

#import "LocalFrame.h"
#import "LocalFrameView.h"
#import "Logging.h"
#import "Page.h"
#import "PlatformCALayerContentsDelayedReleaser.h"
#import "PlatformWheelEvent.h"
#import "Region.h"
#import "ScrollingStateTree.h"
#import "ScrollingThread.h"
#import "ScrollingTreeFixedNodeCocoa.h"
#import "ScrollingTreeFrameScrollingNodeMac.h"
#import "ScrollingTreeMac.h"
#import "ScrollingTreeStickyNodeCocoa.h"
#import "TiledBacking.h"
#import <wtf/MainThread.h>

namespace WebCore {

Ref<ScrollingCoordinator> ScrollingCoordinator::create(Page* page)
{
    return adoptRef(*new ScrollingCoordinatorMac(page));
}

ScrollingCoordinatorMac::ScrollingCoordinatorMac(Page* page)
    : ThreadedScrollingCoordinator(page)
{
    setScrollingTree(ScrollingTreeMac::create(*this));
}

ScrollingCoordinatorMac::~ScrollingCoordinatorMac()
{
    ASSERT(!scrollingTree());
}

void ScrollingCoordinatorMac::commitTreeStateIfNeeded()
{
    ThreadedScrollingCoordinator::commitTreeStateIfNeeded();
    updateTiledScrollingIndicator();
}

void ScrollingCoordinatorMac::willStartPlatformRenderingUpdate()
{
    PlatformCALayerContentsDelayedReleaser::singleton().mainThreadCommitWillStart();
}

void ScrollingCoordinatorMac::didCompletePlatformRenderingUpdate()
{
    downcast<ScrollingTreeMac>(scrollingTree())->didCompletePlatformRenderingUpdate();
    PlatformCALayerContentsDelayedReleaser::singleton().mainThreadCommitDidEnd();
}

void ScrollingCoordinatorMac::updateTiledScrollingIndicator()
{
    RefPtr localMainFrame = page()->localMainFrame();
    if (!localMainFrame)
        return;

    RefPtr frameView = localMainFrame->view();
    if (!frameView)
        return;
    
    CheckedPtr tiledBacking = frameView->tiledBacking();
    if (!tiledBacking)
        return;

    ScrollingModeIndication indicatorMode;
    if (shouldUpdateScrollLayerPositionSynchronously(*frameView))
        indicatorMode = SynchronousScrollingBecauseOfStyleIndication;
    else
        indicatorMode = AsyncScrollingIndication;
    
    tiledBacking->setScrollingModeIndication(indicatorMode);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
