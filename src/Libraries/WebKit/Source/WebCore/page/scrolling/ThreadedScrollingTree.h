/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)

#include "ScrollingStateScrollingNode.h"
#include "ScrollingStateTree.h"
#include "ScrollingTree.h"
#include <wtf/Condition.h>
#include <wtf/RefPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AsyncScrollingCoordinator;

// The ThreadedScrollingTree class lives almost exclusively on the scrolling thread and manages the
// hierarchy of scrollable regions on the page. It's also responsible for dispatching events
// to the correct scrolling tree nodes or dispatching events back to the ScrollingCoordinator
// object on the main thread if they can't be handled on the scrolling thread for various reasons.
class ThreadedScrollingTree : public ScrollingTree {
    WTF_MAKE_TZONE_ALLOCATED(ThreadedScrollingTree);
public:
    virtual ~ThreadedScrollingTree();

    WheelEventHandlingResult handleWheelEvent(const PlatformWheelEvent&, OptionSet<WheelEventProcessingSteps>) override;

    bool handleWheelEventAfterMainThread(const PlatformWheelEvent&, ScrollingNodeID, std::optional<WheelScrollGestureState>);
    void wheelEventWasProcessedByMainThread(const PlatformWheelEvent&, std::optional<WheelScrollGestureState>);

    WEBCORE_EXPORT void willSendEventToMainThread(const PlatformWheelEvent&);
    WEBCORE_EXPORT void waitForEventToBeProcessedByMainThread(const PlatformWheelEvent&);

    void invalidate() override;

    void displayDidRefresh(PlatformDisplayID) override;

    void didScheduleRenderingUpdate();
    void willStartRenderingUpdate();
    virtual void didCompleteRenderingUpdate();

    Lock& treeLock() WTF_RETURNS_LOCK(m_treeLock) { return m_treeLock; }

    bool scrollAnimatorEnabled() const { return m_scrollAnimatorEnabled; }
    void removePendingScrollAnimationForNode(ScrollingNodeID) WTF_REQUIRES_LOCK(m_treeLock) final;

protected:
    explicit ThreadedScrollingTree(AsyncScrollingCoordinator&);

    void scrollingTreeNodeDidScroll(ScrollingTreeScrollingNode&, ScrollingLayerPositionAction = ScrollingLayerPositionAction::Sync) override;
    void scrollingTreeNodeWillStartAnimatedScroll(ScrollingTreeScrollingNode&) override;
    void scrollingTreeNodeDidStopAnimatedScroll(ScrollingTreeScrollingNode&) override;
    void scrollingTreeNodeWillStartWheelEventScroll(ScrollingTreeScrollingNode&) override;
    void scrollingTreeNodeDidStopWheelEventScroll(ScrollingTreeScrollingNode&) override;
    bool scrollingTreeNodeRequestsScroll(ScrollingNodeID, const RequestedScrollData&) override WTF_REQUIRES_LOCK(m_treeLock);
    bool scrollingTreeNodeRequestsKeyboardScroll(ScrollingNodeID, const RequestedKeyboardScrollData&) override WTF_REQUIRES_LOCK(m_treeLock);

#if PLATFORM(MAC)
    void handleWheelEventPhase(ScrollingNodeID, PlatformWheelEventPhase) override;
#endif

    void setActiveScrollSnapIndices(ScrollingNodeID, std::optional<unsigned> horizontalIndex, std::optional<unsigned> verticalIndex) override;

#if PLATFORM(COCOA)
    void currentSnapPointIndicesDidChange(ScrollingNodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical) override;
#endif

    void renderingUpdateComplete();

    void reportExposedUnfilledArea(MonotonicTime, unsigned unfilledArea) override;
    void reportSynchronousScrollingReasonsChanged(MonotonicTime, OptionSet<SynchronousScrollingReason>) override;

    RefPtr<AsyncScrollingCoordinator> m_scrollingCoordinator;
    mutable Lock m_layerHitTestMutex;

private:
    bool isThreadedScrollingTree() const override { return true; }
    void propagateSynchronousScrollingReasons(const UncheckedKeyHashSet<ScrollingNodeID>&) WTF_REQUIRES_LOCK(m_treeLock) override;

    void didCommitTree() final;
    void didCommitTreeOnScrollingThread() WTF_REQUIRES_LOCK(m_treeLock);

    void displayDidRefreshOnScrollingThread();
    void waitForRenderingUpdateCompletionOrTimeout() WTF_REQUIRES_LOCK(m_treeLock);

    bool canUpdateLayersOnScrollingThread() const WTF_REQUIRES_LOCK(m_treeLock);

    void scheduleDelayedRenderingUpdateDetectionTimer(Seconds) WTF_REQUIRES_LOCK(m_treeLock);
    void delayedRenderingUpdateDetectionTimerFired();

    void hasNodeWithAnimatedScrollChanged(bool) final;
    
    bool isScrollingSynchronizedWithMainThread() final WTF_REQUIRES_LOCK(m_treeLock);

    void receivedWheelEventWithPhases(PlatformWheelEventPhase, PlatformWheelEventPhase) final;
    void deferWheelEventTestCompletionForReason(ScrollingNodeID, WheelEventTestMonitor::DeferReason) final;
    void removeWheelEventTestCompletionDeferralForReason(ScrollingNodeID, WheelEventTestMonitor::DeferReason) final;

    void lockLayersForHitTesting() final WTF_ACQUIRES_LOCK(m_layerHitTestMutex);
    void unlockLayersForHitTesting() final WTF_RELEASES_LOCK(m_layerHitTestMutex);

    void scrollingTreeNodeScrollUpdated(ScrollingTreeScrollingNode&, const ScrollUpdateType&);

    enum class SynchronizationState : uint8_t {
        Idle,
        WaitingForRenderingUpdate,
        InRenderingUpdate,
        Desynchronized,
    };

    SynchronizationState m_state WTF_GUARDED_BY_LOCK(m_treeLock) { SynchronizationState::Idle };
    Condition m_stateCondition;

    bool m_receivedBeganEventFromMainThread WTF_GUARDED_BY_LOCK(m_treeLock) { false };
    Condition m_waitingForBeganEventCondition;
    
    MonotonicTime m_lastDisplayDidRefreshTime;

    // Dynamically allocated because it has to use the ScrollingThread's runloop.
    std::unique_ptr<RunLoop::Timer> m_delayedRenderingUpdateDetectionTimer WTF_GUARDED_BY_LOCK(m_treeLock);

    UncheckedKeyHashMap<ScrollingNodeID, RequestedScrollData> m_nodesWithPendingScrollAnimations WTF_GUARDED_BY_LOCK(m_treeLock);
    UncheckedKeyHashMap<ScrollingNodeID, RequestedKeyboardScrollData> m_nodesWithPendingKeyboardScrollAnimations WTF_GUARDED_BY_LOCK(m_treeLock);
    const bool m_scrollAnimatorEnabled { false };
    bool m_hasNodesWithSynchronousScrollingReasons WTF_GUARDED_BY_LOCK(m_treeLock) { false };
    std::atomic<bool> m_renderingUpdateWasScheduled;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_TREE(WebCore::ThreadedScrollingTree, isThreadedScrollingTree())

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
