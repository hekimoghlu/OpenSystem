/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#include "RemoteScrollingTree.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(UI_SIDE_COMPOSITING)

#include "RemoteLayerTreeHost.h"
#include "RemoteScrollingCoordinatorProxy.h"
#include <WebCore/ScrollingTreeFixedNodeCocoa.h>
#include <WebCore/ScrollingTreeFrameHostingNode.h>
#include <WebCore/ScrollingTreeFrameScrollingNode.h>
#include <WebCore/ScrollingTreeOverflowScrollProxyNodeCocoa.h>
#include <WebCore/ScrollingTreePluginHostingNode.h>
#include <WebCore/ScrollingTreePluginScrollingNode.h>
#include <WebCore/ScrollingTreePositionedNodeCocoa.h>
#include <WebCore/ScrollingTreeStickyNodeCocoa.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteScrollingTree);

using namespace WebCore;

RemoteScrollingTree::RemoteScrollingTree(RemoteScrollingCoordinatorProxy& scrollingCoordinator)
    : m_scrollingCoordinatorProxy(WeakPtr { scrollingCoordinator })
{
}

RemoteScrollingTree::~RemoteScrollingTree() = default;

void RemoteScrollingTree::invalidate()
{
    ASSERT(isMainRunLoop());
    Locker locker { m_treeLock };
    removeAllNodes();
    m_scrollingCoordinatorProxy = nullptr;
}

RemoteScrollingCoordinatorProxy* RemoteScrollingTree::scrollingCoordinatorProxy() const
{
    ASSERT(isMainRunLoop());
    return m_scrollingCoordinatorProxy.get();
}

void RemoteScrollingTree::scrollingTreeNodeDidScroll(ScrollingTreeScrollingNode& node, ScrollingLayerPositionAction scrollingLayerPositionAction)
{
    ASSERT(isMainRunLoop());

    ScrollingTree::scrollingTreeNodeDidScroll(node, scrollingLayerPositionAction);

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return;

    std::optional<FloatPoint> layoutViewportOrigin;
    if (auto* scrollingNode = dynamicDowncast<ScrollingTreeFrameScrollingNode>(node))
        layoutViewportOrigin = scrollingNode->layoutViewport().location();

    auto scrollUpdate = ScrollUpdate { node.scrollingNodeID(), node.currentScrollPosition(), layoutViewportOrigin, ScrollUpdateType::PositionUpdate, scrollingLayerPositionAction };
    addPendingScrollUpdate(WTFMove(scrollUpdate));

    scrollingCoordinatorProxy->scrollingThreadAddedPendingUpdate();
}

void RemoteScrollingTree::scrollingTreeNodeDidStopAnimatedScroll(ScrollingTreeScrollingNode& node)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return;

    scrollingCoordinatorProxy->scrollingTreeNodeDidStopAnimatedScroll(node.scrollingNodeID());
}

bool RemoteScrollingTree::scrollingTreeNodeRequestsScroll(ScrollingNodeID nodeID, const RequestedScrollData& request)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return false;

    return scrollingCoordinatorProxy->scrollingTreeNodeRequestsScroll(nodeID, request);
}

bool RemoteScrollingTree::scrollingTreeNodeRequestsKeyboardScroll(ScrollingNodeID nodeID, const RequestedKeyboardScrollData& request)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return false;

    return scrollingCoordinatorProxy->scrollingTreeNodeRequestsKeyboardScroll(nodeID, request);
}

void RemoteScrollingTree::scrollingTreeNodeWillStartScroll(ScrollingNodeID nodeID)
{
    if (CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get())
        scrollingCoordinatorProxy->scrollingTreeNodeWillStartScroll(nodeID);
}

void RemoteScrollingTree::scrollingTreeNodeDidEndScroll(ScrollingNodeID nodeID)
{
    if (CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get())
        scrollingCoordinatorProxy->scrollingTreeNodeDidEndScroll(nodeID);
}

void RemoteScrollingTree::clearNodesWithUserScrollInProgress()
{
    ScrollingTree::clearNodesWithUserScrollInProgress();

    if (CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get())
        scrollingCoordinatorProxy->clearNodesWithUserScrollInProgress();
}

void RemoteScrollingTree::scrollingTreeNodeDidBeginScrollSnapping(ScrollingNodeID nodeID)
{
    if (CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get())
        scrollingCoordinatorProxy->scrollingTreeNodeDidBeginScrollSnapping(nodeID);
}

void RemoteScrollingTree::scrollingTreeNodeDidEndScrollSnapping(ScrollingNodeID nodeID)
{
    if (CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get())
        scrollingCoordinatorProxy->scrollingTreeNodeDidEndScrollSnapping(nodeID);
}

Ref<ScrollingTreeNode> RemoteScrollingTree::createScrollingTreeNode(ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    switch (nodeType) {
    case ScrollingNodeType::MainFrame:
    case ScrollingNodeType::Subframe:
    case ScrollingNodeType::Overflow:
    case ScrollingNodeType::PluginScrolling:
        ASSERT_NOT_REACHED(); // Subclass should have handled this.
        break;

    case ScrollingNodeType::FrameHosting:
        return ScrollingTreeFrameHostingNode::create(*this, nodeID);
    case ScrollingNodeType::PluginHosting:
        return ScrollingTreePluginHostingNode::create(*this, nodeID);
    case ScrollingNodeType::OverflowProxy:
        return ScrollingTreeOverflowScrollProxyNodeCocoa::create(*this, nodeID);
    case ScrollingNodeType::Fixed:
        return ScrollingTreeFixedNodeCocoa::create(*this, nodeID);
    case ScrollingNodeType::Sticky:
        return ScrollingTreeStickyNodeCocoa::create(*this, nodeID);
    case ScrollingNodeType::Positioned:
        return ScrollingTreePositionedNodeCocoa::create(*this, nodeID);
    }
    ASSERT_NOT_REACHED();
    return ScrollingTreeFixedNodeCocoa::create(*this, nodeID);
}

void RemoteScrollingTree::currentSnapPointIndicesDidChange(ScrollingNodeID nodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return;

    scrollingCoordinatorProxy->currentSnapPointIndicesDidChange(nodeID, horizontal, vertical);
}

void RemoteScrollingTree::reportExposedUnfilledArea(MonotonicTime time, unsigned unfilledArea)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return;

    scrollingCoordinatorProxy->reportExposedUnfilledArea(time, unfilledArea);
}

void RemoteScrollingTree::reportSynchronousScrollingReasonsChanged(MonotonicTime timestamp, OptionSet<SynchronousScrollingReason> reasons)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return;

    scrollingCoordinatorProxy->reportSynchronousScrollingReasonsChanged(timestamp, reasons);
}

void RemoteScrollingTree::receivedWheelEventWithPhases(PlatformWheelEventPhase phase, PlatformWheelEventPhase momentumPhase)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy)
        return;

    scrollingCoordinatorProxy->receivedWheelEventWithPhases(phase, momentumPhase);
}

void RemoteScrollingTree::deferWheelEventTestCompletionForReason(ScrollingNodeID nodeID, WheelEventTestMonitor::DeferReason reason)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy || !isMonitoringWheelEvents())
        return;

    scrollingCoordinatorProxy->deferWheelEventTestCompletionForReason(nodeID, reason);
}

void RemoteScrollingTree::removeWheelEventTestCompletionDeferralForReason(ScrollingNodeID nodeID, WheelEventTestMonitor::DeferReason reason)
{
    ASSERT(isMainRunLoop());

    CheckedPtr scrollingCoordinatorProxy = m_scrollingCoordinatorProxy.get();
    if (!scrollingCoordinatorProxy || !isMonitoringWheelEvents())
        return;

    scrollingCoordinatorProxy->removeWheelEventTestCompletionDeferralForReason(nodeID, reason);
}

void RemoteScrollingTree::propagateSynchronousScrollingReasons(const UncheckedKeyHashSet<ScrollingNodeID>& synchronousScrollingNodes)
{
    m_hasNodesWithSynchronousScrollingReasons = !synchronousScrollingNodes.isEmpty();
}

void RemoteScrollingTree::tryToApplyLayerPositions()
{
    ASSERT(!isMainRunLoop());
    Locker locker { m_treeLock };
    if (m_hasNodesWithSynchronousScrollingReasons)
        return;

    applyLayerPositionsInternal();
}


} // namespace WebKit

#endif // ENABLE(UI_SIDE_COMPOSITING)
