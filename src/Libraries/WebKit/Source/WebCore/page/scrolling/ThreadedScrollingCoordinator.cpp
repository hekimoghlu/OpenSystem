/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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
#include "ThreadedScrollingCoordinator.h"

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
#include "Logging.h"
#include "Page.h"
#include "ScrollingThread.h"
#include "ThreadedScrollingTree.h"
#include "WheelEventTestMonitor.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ThreadedScrollingCoordinator);

ThreadedScrollingCoordinator::ThreadedScrollingCoordinator(Page* page)
    : AsyncScrollingCoordinator(page)
{
}

ThreadedScrollingCoordinator::~ThreadedScrollingCoordinator() = default;

void ThreadedScrollingCoordinator::pageDestroyed()
{
    AsyncScrollingCoordinator::pageDestroyed();

    // Invalidating the scrolling tree will break the reference cycle between the ScrollingCoordinator and ScrollingTree objects.
    RefPtr scrollingTree = static_pointer_cast<ThreadedScrollingTree>(releaseScrollingTree());
    ScrollingThread::dispatch([scrollingTree = WTFMove(scrollingTree)] {
        scrollingTree->invalidate();
    });
}

void ThreadedScrollingCoordinator::commitTreeStateIfNeeded()
{
    scrollingStateTrees().forEach([&] (auto& key, auto& value) {
        willCommitTree(key);

        LOG_WITH_STREAM(Scrolling, stream << "ThreadedScrollingCoordinator::commitTreeState, has changes " << value->hasChangedProperties());

        if (!value->hasChangedProperties())
            return;

        LOG_WITH_STREAM(ScrollingTree, stream << "ThreadedScrollingCoordinator::commitTreeState: state tree " << scrollingStateTreeAsText(debugScrollingStateTreeAsTextBehaviors));

        auto stateTree = commitTreeStateForRootFrameID(key, LayerRepresentation::PlatformLayerRepresentation);
        stateTree->setRootFrameIdentifier(key);
        scrollingTree()->commitTreeState(WTFMove(stateTree));
    });
}

WheelEventHandlingResult ThreadedScrollingCoordinator::handleWheelEventForScrolling(const PlatformWheelEvent& wheelEvent, ScrollingNodeID targetNodeID, std::optional<WheelScrollGestureState> gestureState)
{
    ASSERT(isMainThread());
    ASSERT(page());
    ASSERT(scrollingTree());

    if (scrollingTree()->willWheelEventStartSwipeGesture(wheelEvent))
        return WheelEventHandlingResult::unhandled();

    LOG_WITH_STREAM(Scrolling, stream << "ThreadedScrollingCoordinator::handleWheelEventForScrolling " << wheelEvent << " - sending event to scrolling thread, node " << targetNodeID << " gestureState " << gestureState);

    auto deferrer = WheelEventTestMonitorCompletionDeferrer { page()->wheelEventTestMonitor().get(), targetNodeID, WheelEventTestMonitor::DeferReason::PostMainThreadWheelEventHandling };

    RefPtr<ThreadedScrollingTree> threadedScrollingTree = downcast<ThreadedScrollingTree>(scrollingTree());
    ScrollingThread::dispatch([threadedScrollingTree, wheelEvent, targetNodeID, gestureState, deferrer = WTFMove(deferrer)] {
        threadedScrollingTree->handleWheelEventAfterMainThread(wheelEvent, targetNodeID, gestureState);
    });
    return WheelEventHandlingResult::handled();
}

void ThreadedScrollingCoordinator::wheelEventWasProcessedByMainThread(const PlatformWheelEvent& wheelEvent, std::optional<WheelScrollGestureState> gestureState)
{
    LOG_WITH_STREAM(Scrolling, stream << "ThreadedScrollingCoordinator::wheelEventWasProcessedByMainThread " << gestureState);

    RefPtr<ThreadedScrollingTree> threadedScrollingTree = downcast<ThreadedScrollingTree>(scrollingTree());
    threadedScrollingTree->wheelEventWasProcessedByMainThread(wheelEvent, gestureState);
}

void ThreadedScrollingCoordinator::scheduleTreeStateCommit()
{
    scheduleRenderingUpdate();
}

void ThreadedScrollingCoordinator::didScheduleRenderingUpdate()
{
    RefPtr<ThreadedScrollingTree> threadedScrollingTree = downcast<ThreadedScrollingTree>(scrollingTree());
    threadedScrollingTree->didScheduleRenderingUpdate();
}

void ThreadedScrollingCoordinator::willStartRenderingUpdate()
{
    ASSERT(isMainThread());
    if (RefPtr page = this->page())
        page->layoutIfNeeded(LayoutOptions::UpdateCompositingLayers);
    RefPtr threadedScrollingTree = downcast<ThreadedScrollingTree>(scrollingTree());
    threadedScrollingTree->willStartRenderingUpdate();
    commitTreeStateIfNeeded();
    synchronizeStateFromScrollingTree();
}

void ThreadedScrollingCoordinator::didCompleteRenderingUpdate()
{
    downcast<ThreadedScrollingTree>(scrollingTree())->didCompleteRenderingUpdate();

    // When scroll animations are running on the scrolling thread, we need something to continually tickle the
    // DisplayRefreshMonitor so that ThreadedScrollingTree::displayDidRefresh() will get called to service those animations.
    // We can achieve this by scheduling a rendering update; this won't cause extra work, since scrolling thread scrolls
    // will end up triggering these anyway.
    if (scrollingTree()->hasNodeWithActiveScrollAnimations())
        scheduleRenderingUpdate();
}

void ThreadedScrollingCoordinator::hasNodeWithAnimatedScrollChanged(bool hasAnimatingNode)
{
    // This is necessary to trigger a rendering update, after which the code in
    // AsyncScrollingCoordinator::didCompleteRenderingUpdate() triggers the rest.
    if (hasAnimatingNode)
        scheduleRenderingUpdate();
}

void ThreadedScrollingCoordinator::startMonitoringWheelEvents(bool clearLatchingState)
{
    // setIsMonitoringWheelEvents() is called on the root node via AsyncScrollingCoordinator::frameViewLayoutUpdated().

    if (clearLatchingState)
        scrollingTree()->clearLatchedNode();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
