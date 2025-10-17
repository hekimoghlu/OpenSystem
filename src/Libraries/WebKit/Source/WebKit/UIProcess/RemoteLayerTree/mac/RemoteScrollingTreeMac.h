/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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

#include "RemoteScrollingTree.h"

#if PLATFORM(MAC) && ENABLE(UI_SIDE_COMPOSITING)

#include "RemoteScrollingCoordinator.h"
#include <WebCore/ScrollingConstraints.h>
#include <WebCore/ScrollingTree.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class PlatformMouseEvent;
};

namespace WebKit {

class RemoteScrollingCoordinatorProxy;

class RemoteScrollingTreeMac final : public RemoteScrollingTree {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingTreeMac);
public:
    explicit RemoteScrollingTreeMac(RemoteScrollingCoordinatorProxy&);
    virtual ~RemoteScrollingTreeMac();

    void scrollingTreeNodeScrollbarVisibilityDidChange(WebCore::ScrollingNodeID, WebCore::ScrollbarOrientation, bool) override;
    void scrollingTreeNodeScrollbarMinimumThumbLengthDidChange(WebCore::ScrollingNodeID, WebCore::ScrollbarOrientation, int) override;

private:
    void handleWheelEventPhase(WebCore::ScrollingNodeID, WebCore::PlatformWheelEventPhase) override;
    RefPtr<WebCore::ScrollingTreeNode> scrollingNodeForPoint(WebCore::FloatPoint) override;
#if ENABLE(WHEEL_EVENT_REGIONS)
    OptionSet<WebCore::EventListenerRegionType> eventListenerRegionTypesForPoint(WebCore::FloatPoint) const override;
#endif

    void didCommitTree() override WTF_REQUIRES_LOCK(m_treeLock);

    void scrollingTreeNodeDidScroll(WebCore::ScrollingTreeScrollingNode&, WebCore::ScrollingLayerPositionAction) override;
    void scrollingTreeNodeDidStopAnimatedScroll(WebCore::ScrollingTreeScrollingNode&) override;
    bool scrollingTreeNodeRequestsScroll(WebCore::ScrollingNodeID, const WebCore::RequestedScrollData&) override;
    bool scrollingTreeNodeRequestsKeyboardScroll(WebCore::ScrollingNodeID, const WebCore::RequestedKeyboardScrollData&) override;
    void currentSnapPointIndicesDidChange(WebCore::ScrollingNodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical) override;
    void reportExposedUnfilledArea(MonotonicTime, unsigned unfilledArea) override;
    void reportSynchronousScrollingReasonsChanged(MonotonicTime, OptionSet<WebCore::SynchronousScrollingReason>) override;
    void receivedWheelEventWithPhases(WebCore::PlatformWheelEventPhase, WebCore::PlatformWheelEventPhase momentumPhase) override;

    // "Default handling" here refers to sending the event to the web process for synchronous scrolling, and DOM event handling.
    void willSendEventForDefaultHandling(const WebCore::PlatformWheelEvent&) override;
    void waitForEventDefaultHandlingCompletion(const WebCore::PlatformWheelEvent&) override;
    void receivedEventAfterDefaultHandling(const WebCore::PlatformWheelEvent&, std::optional<WebCore::WheelScrollGestureState>) override;

    WebCore::WheelEventHandlingResult handleWheelEventAfterDefaultHandling(const WebCore::PlatformWheelEvent&, std::optional<WebCore::ScrollingNodeID>, std::optional<WebCore::WheelScrollGestureState>) override;

    void deferWheelEventTestCompletionForReason(WebCore::ScrollingNodeID, WebCore::WheelEventTestMonitor::DeferReason) override;
    void removeWheelEventTestCompletionDeferralForReason(WebCore::ScrollingNodeID, WebCore::WheelEventTestMonitor::DeferReason) override;

    void hasNodeWithAnimatedScrollChanged(bool) override;
    void setRubberBandingInProgressForNode(WebCore::ScrollingNodeID, bool) override;

    void displayDidRefresh(WebCore::PlatformDisplayID) override;

    void lockLayersForHitTesting() final WTF_ACQUIRES_LOCK(m_layerHitTestMutex);
    void unlockLayersForHitTesting() final WTF_RELEASES_LOCK(m_layerHitTestMutex);

    void startPendingScrollAnimations() WTF_REQUIRES_LOCK(m_treeLock);

    void scrollingTreeNodeWillStartScroll(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScroll(WebCore::ScrollingNodeID) override;
    void clearNodesWithUserScrollInProgress() override;

    void scrollingTreeNodeDidBeginScrollSnapping(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScrollSnapping(WebCore::ScrollingNodeID) override;

    Ref<WebCore::ScrollingTreeNode> createScrollingTreeNode(WebCore::ScrollingNodeType, WebCore::ScrollingNodeID) override;

    HashMap<WebCore::ScrollingNodeID, WebCore::RequestedScrollData> m_nodesWithPendingScrollAnimations; // Guarded by m_treeLock but used via call chains that can't be annotated.
    HashMap<WebCore::ScrollingNodeID, WebCore::RequestedKeyboardScrollData> m_nodesWithPendingKeyboardScrollAnimations; // Guarded by m_treeLock but used via call chains that can't be annotated.

    mutable Lock m_layerHitTestMutex;

    Condition m_waitingForBeganEventCondition;
    bool m_receivedBeganEventFromMainThread WTF_GUARDED_BY_LOCK(m_treeLock) { false };
};

} // namespace WebKit

#endif // PLATFORM(MAC) && ENABLE(UI_SIDE_COMPOSITING)
