/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#import "RemoteScrollingCoordinator.h"

#if ENABLE(ASYNC_SCROLLING)

#import "ArgumentCoders.h"
#import "GraphicsLayerCARemote.h"
#import "Logging.h"
#import "RemoteLayerTreeDrawingArea.h"
#import "RemoteScrollingCoordinatorMessages.h"
#import "RemoteScrollingCoordinatorTransaction.h"
#import "RemoteScrollingUIState.h"
#import "WebPage.h"
#import "WebProcess.h"
#import <WebCore/GraphicsLayer.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Page.h>
#import <WebCore/RenderLayerCompositor.h>
#import <WebCore/RenderView.h>
#import <WebCore/ScrollbarsController.h>
#import <WebCore/ScrollingStateFrameScrollingNode.h>
#import <WebCore/ScrollingStateTree.h>
#import <WebCore/ScrollingTreeFixedNodeCocoa.h>
#import <WebCore/ScrollingTreeStickyNodeCocoa.h>
#import <WebCore/WheelEventTestMonitor.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteScrollingCoordinator);

RemoteScrollingCoordinator::RemoteScrollingCoordinator(WebPage* page)
    : AsyncScrollingCoordinator(page->corePage())
    , m_webPage(page)
{
    WebProcess::singleton().addMessageReceiver(Messages::RemoteScrollingCoordinator::messageReceiverName(), m_webPage->identifier(), *this);
}

RemoteScrollingCoordinator::~RemoteScrollingCoordinator()
{
    WebProcess::singleton().removeMessageReceiver(Messages::RemoteScrollingCoordinator::messageReceiverName(), m_webPage->identifier());
}

void RemoteScrollingCoordinator::scheduleTreeStateCommit()
{
    m_webPage->drawingArea()->triggerRenderingUpdate();
}

bool RemoteScrollingCoordinator::coordinatesScrollingForFrameView(const LocalFrameView& frameView) const
{
    RenderView* renderView = frameView.renderView();
    return renderView && renderView->usesCompositing();
}

bool RemoteScrollingCoordinator::isRubberBandInProgress(std::optional<ScrollingNodeID> nodeID) const
{
    if (!nodeID)
        return false;
    return m_nodesWithActiveRubberBanding.contains(*nodeID);
}

bool RemoteScrollingCoordinator::isUserScrollInProgress(std::optional<ScrollingNodeID> nodeID) const
{
    if (!nodeID)
        return false;
    return m_nodesWithActiveUserScrolls.contains(*nodeID);
}

bool RemoteScrollingCoordinator::isScrollSnapInProgress(std::optional<ScrollingNodeID> nodeID) const
{
    if (!nodeID)
        return false;
    return m_nodesWithActiveScrollSnap.contains(*nodeID);
}

void RemoteScrollingCoordinator::setScrollPinningBehavior(ScrollPinningBehavior)
{
    // FIXME: send to the UI process.
}

RemoteScrollingCoordinatorTransaction RemoteScrollingCoordinator::buildTransaction(FrameIdentifier rootFrameID)
{
    willCommitTree(rootFrameID);

    return {
        ensureScrollingStateTreeForRootFrameID(rootFrameID).commit(LayerRepresentation::PlatformLayerIDRepresentation),
        std::exchange(m_clearScrollLatchingInNextTransaction, false),
        { },
        RemoteScrollingCoordinatorTransaction::FromDeserialization::No
    };
}

// Notification from the UI process that we scrolled.
void RemoteScrollingCoordinator::scrollPositionChangedForNode(ScrollingNodeID nodeID, const FloatPoint& scrollPosition, std::optional<FloatPoint> layoutViewportOrigin, bool syncLayerPosition, CompletionHandler<void()>&& completionHandler)
{
    LOG_WITH_STREAM(Scrolling, stream << "RemoteScrollingCoordinator::scrollingTreeNodeDidScroll " << nodeID << " to " << scrollPosition << " layoutViewportOrigin " << layoutViewportOrigin);

    auto scrollUpdate = ScrollUpdate { nodeID, scrollPosition, layoutViewportOrigin, ScrollUpdateType::PositionUpdate, syncLayerPosition ? ScrollingLayerPositionAction::Sync : ScrollingLayerPositionAction::Set };
    applyScrollUpdate(WTFMove(scrollUpdate));

    completionHandler();
}

void RemoteScrollingCoordinator::animatedScrollDidEndForNode(ScrollingNodeID nodeID)
{
    auto scrollUpdate = ScrollUpdate { nodeID, { }, { }, ScrollUpdateType::AnimatedScrollDidEnd };
    applyScrollUpdate(WTFMove(scrollUpdate));
}

void RemoteScrollingCoordinator::currentSnapPointIndicesChangedForNode(ScrollingNodeID nodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical)
{
    setActiveScrollSnapIndices(nodeID, horizontal, vertical);
}

void RemoteScrollingCoordinator::scrollingStateInUIProcessChanged(const RemoteScrollingUIState& uiState)
{
    // FIXME: Also track m_nodesWithActiveRubberBanding.
    if (uiState.changes().contains(RemoteScrollingUIStateChanges::ScrollSnapNodes))
        m_nodesWithActiveScrollSnap = uiState.nodesWithActiveScrollSnap();

    if (uiState.changes().contains(RemoteScrollingUIStateChanges::UserScrollNodes))
        m_nodesWithActiveUserScrolls = uiState.nodesWithActiveUserScrolls();

    if (uiState.changes().contains(RemoteScrollingUIStateChanges::RubberbandingNodes))
        m_nodesWithActiveRubberBanding = uiState.nodesWithActiveRubberband();
}

void RemoteScrollingCoordinator::addNodeWithActiveRubberBanding(ScrollingNodeID nodeID)
{
    m_nodesWithActiveRubberBanding.add(nodeID);
}

void RemoteScrollingCoordinator::removeNodeWithActiveRubberBanding(ScrollingNodeID nodeID)
{
    m_nodesWithActiveRubberBanding.remove(nodeID);
}

void RemoteScrollingCoordinator::startMonitoringWheelEvents(bool clearLatchingState)
{
    if (clearLatchingState)
        m_clearScrollLatchingInNextTransaction = true;
}

void RemoteScrollingCoordinator::receivedWheelEventWithPhases(WebCore::PlatformWheelEventPhase phase, WebCore::PlatformWheelEventPhase momentumPhase)
{
    if (auto monitor = page()->wheelEventTestMonitor())
        monitor->receivedWheelEventWithPhases(phase, momentumPhase);
}

void RemoteScrollingCoordinator::startDeferringScrollingTestCompletionForNode(WebCore::ScrollingNodeID nodeID, OptionSet<WebCore::WheelEventTestMonitor::DeferReason> reason)
{
    if (auto monitor = page()->wheelEventTestMonitor())
        monitor->deferForReason(nodeID, reason);
}

void RemoteScrollingCoordinator::stopDeferringScrollingTestCompletionForNode(WebCore::ScrollingNodeID nodeID, OptionSet<WebCore::WheelEventTestMonitor::DeferReason> reason)
{
    if (auto monitor = page()->wheelEventTestMonitor())
        monitor->removeDeferralForReason(nodeID, reason);
}

WheelEventHandlingResult RemoteScrollingCoordinator::handleWheelEventForScrolling(const PlatformWheelEvent& wheelEvent, ScrollingNodeID targetNodeID, std::optional<WheelScrollGestureState> gestureState)
{
    LOG_WITH_STREAM(Scrolling, stream << "RemoteScrollingCoordinator::handleWheelEventForScrolling " << wheelEvent << " - node " << targetNodeID << " gestureState " << gestureState << " will start swipe " << (m_currentWheelEventWillStartSwipe && *m_currentWheelEventWillStartSwipe));

    if (m_currentWheelEventWillStartSwipe && *m_currentWheelEventWillStartSwipe)
        return WheelEventHandlingResult::unhandled();

    m_currentWheelGestureInfo = NodeAndGestureState { targetNodeID, gestureState };
    return WheelEventHandlingResult::handled();
}

void RemoteScrollingCoordinator::scrollingTreeNodeScrollbarVisibilityDidChange(ScrollingNodeID nodeID, ScrollbarOrientation orientation, bool isVisible)
{
    auto* frameView = frameViewForScrollingNode(nodeID);
    if (!frameView)
        return;

    if (auto* scrollableArea = frameView->scrollableAreaForScrollingNodeID(nodeID))
        scrollableArea->scrollbarsController().setScrollbarVisibilityState(orientation, isVisible);
}

void RemoteScrollingCoordinator::scrollingTreeNodeScrollbarMinimumThumbLengthDidChange(ScrollingNodeID nodeID, ScrollbarOrientation orientation, int minimumThumbLength)
{
    auto* frameView = frameViewForScrollingNode(nodeID);
    if (!frameView)
        return;

    if (auto* scrollableArea = frameView->scrollableAreaForScrollingNodeID(nodeID))
        scrollableArea->scrollbarsController().setScrollbarMinimumThumbLength(orientation, minimumThumbLength);
}

} // namespace WebKit

#endif // ENABLE(ASYNC_SCROLLING)
