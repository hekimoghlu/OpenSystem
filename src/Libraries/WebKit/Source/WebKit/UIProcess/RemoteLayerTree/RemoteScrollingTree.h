/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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

#if ENABLE(UI_SIDE_COMPOSITING)

#include <WebCore/ScrollingConstraints.h>
#include <WebCore/ScrollingCoordinatorTypes.h>
#include <WebCore/ScrollingTree.h>
#include <WebCore/WheelEventTestMonitor.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class PlatformMouseEvent;
};

namespace WebKit {

class RemoteScrollingCoordinatorProxy;

class RemoteScrollingTree : public WebCore::ScrollingTree {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingTree);
public:
    static Ref<RemoteScrollingTree> create(RemoteScrollingCoordinatorProxy&);
    virtual ~RemoteScrollingTree();

    bool isRemoteScrollingTree() const final { return true; }

    void invalidate() final;

    virtual void willSendEventForDefaultHandling(const WebCore::PlatformWheelEvent&) { }
    virtual void waitForEventDefaultHandlingCompletion(const WebCore::PlatformWheelEvent&) { }
    virtual void receivedEventAfterDefaultHandling(const WebCore::PlatformWheelEvent&, std::optional<WebCore::WheelScrollGestureState>) { };
    virtual WebCore::WheelEventHandlingResult handleWheelEventAfterDefaultHandling(const WebCore::PlatformWheelEvent&, std::optional<WebCore::ScrollingNodeID>, std::optional<WebCore::WheelScrollGestureState>) { return WebCore::WheelEventHandlingResult::unhandled(); }

    RemoteScrollingCoordinatorProxy* scrollingCoordinatorProxy() const;

    void scrollingTreeNodeDidScroll(WebCore::ScrollingTreeScrollingNode&, WebCore::ScrollingLayerPositionAction = WebCore::ScrollingLayerPositionAction::Sync) override;
    void scrollingTreeNodeDidStopAnimatedScroll(WebCore::ScrollingTreeScrollingNode&) override;
    bool scrollingTreeNodeRequestsScroll(WebCore::ScrollingNodeID, const WebCore::RequestedScrollData&) override;
    bool scrollingTreeNodeRequestsKeyboardScroll(WebCore::ScrollingNodeID, const WebCore::RequestedKeyboardScrollData&) override;

    void scrollingTreeNodeWillStartScroll(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScroll(WebCore::ScrollingNodeID) override;
    void clearNodesWithUserScrollInProgress() override;

    void scrollingTreeNodeDidBeginScrollSnapping(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScrollSnapping(WebCore::ScrollingNodeID) override;

    void currentSnapPointIndicesDidChange(WebCore::ScrollingNodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical) override;
    void reportExposedUnfilledArea(MonotonicTime, unsigned unfilledArea) override;
    void reportSynchronousScrollingReasonsChanged(MonotonicTime, OptionSet<WebCore::SynchronousScrollingReason>) override;

    void tryToApplyLayerPositions();

protected:
    explicit RemoteScrollingTree(RemoteScrollingCoordinatorProxy&);

    Ref<WebCore::ScrollingTreeNode> createScrollingTreeNode(WebCore::ScrollingNodeType, WebCore::ScrollingNodeID) override;

    void receivedWheelEventWithPhases(WebCore::PlatformWheelEventPhase phase, WebCore::PlatformWheelEventPhase momentumPhase) override;
    void deferWheelEventTestCompletionForReason(WebCore::ScrollingNodeID, WebCore::WheelEventTestMonitor::DeferReason) override;
    void removeWheelEventTestCompletionDeferralForReason(WebCore::ScrollingNodeID, WebCore::WheelEventTestMonitor::DeferReason) override;
    void propagateSynchronousScrollingReasons(const UncheckedKeyHashSet<WebCore::ScrollingNodeID>&) WTF_REQUIRES_LOCK(m_treeLock) override;

    // This gets nulled out via invalidate(), since the scrolling thread can hold a ref to the ScrollingTree after the RemoteScrollingCoordinatorProxy has gone away.
    WeakPtr<RemoteScrollingCoordinatorProxy> m_scrollingCoordinatorProxy;
    bool m_hasNodesWithSynchronousScrollingReasons WTF_GUARDED_BY_LOCK(m_treeLock) { false };
};

class RemoteLayerTreeHitTestLocker {
public:
    RemoteLayerTreeHitTestLocker(RemoteScrollingTree& scrollingTree)
        : m_scrollingTree(scrollingTree)
    {
        m_scrollingTree->lockLayersForHitTesting();
    }
    
    ~RemoteLayerTreeHitTestLocker()
    {
        m_scrollingTree->unlockLayersForHitTesting();
    }

private:
    Ref<RemoteScrollingTree> m_scrollingTree;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_SCROLLING_TREE(WebKit::RemoteScrollingTree, isRemoteScrollingTree());

#endif // ENABLE(UI_SIDE_COMPOSITING)
