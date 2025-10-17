/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#import "RemoteScrollingCoordinatorProxyMac.h"

#if PLATFORM(MAC) && ENABLE(UI_SIDE_COMPOSITING)

#import "RemoteLayerTreeDrawingAreaProxy.h"
#import "RemoteLayerTreeEventDispatcher.h"
#import "WebPageProxy.h"
#import <WebCore/PerformanceLoggingClient.h>
#import <WebCore/ScrollingStateFrameScrollingNode.h>
#import <WebCore/ScrollingStateOverflowScrollProxyNode.h>
#import <WebCore/ScrollingStateOverflowScrollingNode.h>
#import <WebCore/ScrollingStatePluginScrollingNode.h>
#import <WebCore/ScrollingStatePositionedNode.h>
#import <WebCore/ScrollingStateTree.h>
#import <WebCore/ScrollingTreeFrameScrollingNode.h>
#import <WebCore/ScrollingTreeOverflowScrollProxyNode.h>
#import <WebCore/ScrollingTreeOverflowScrollingNode.h>
#import <WebCore/ScrollingTreePluginScrollingNode.h>
#import <WebCore/ScrollingTreePositionedNode.h>
#import <WebCore/WheelEventDeltaFilter.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteScrollingCoordinatorProxyMac);

using namespace WebCore;

RemoteScrollingCoordinatorProxyMac::RemoteScrollingCoordinatorProxyMac(WebPageProxy& webPageProxy)
    : RemoteScrollingCoordinatorProxy(webPageProxy)
#if ENABLE(SCROLLING_THREAD)
    , m_eventDispatcher(RemoteLayerTreeEventDispatcher::create(*this, webPageProxy.webPageIDInMainFrameProcess()))
#endif
{
    m_eventDispatcher->setScrollingTree(scrollingTree());
}

RemoteScrollingCoordinatorProxyMac::~RemoteScrollingCoordinatorProxyMac()
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->invalidate();
#endif
}

void RemoteScrollingCoordinatorProxyMac::cacheWheelEventScrollingAccelerationCurve(const NativeWebWheelEvent& nativeWheelEvent)
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->cacheWheelEventScrollingAccelerationCurve(nativeWheelEvent);
#else
    UNUSED_PARAM(nativeWheelEvent);
#endif
}

void RemoteScrollingCoordinatorProxyMac::handleWheelEvent(const WebWheelEvent& wheelEvent, RectEdges<bool> rubberBandableEdges)
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->handleWheelEvent(wheelEvent, rubberBandableEdges);
#else
    UNUSED_PARAM(wheelEvent);
    UNUSED_PARAM(rubberBandableEdges);
#endif
}

void RemoteScrollingCoordinatorProxyMac::wheelEventHandlingCompleted(const PlatformWheelEvent& wheelEvent, std::optional<ScrollingNodeID> scrollingNodeID, std::optional<WheelScrollGestureState> gestureState, bool wasHandled)
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->wheelEventHandlingCompleted(wheelEvent, scrollingNodeID, gestureState, wasHandled);
#else
    UNUSED_PARAM(wheelEvent);
    UNUSED_PARAM(scrollingNodeID);
    UNUSED_PARAM(gestureState);
    UNUSED_PARAM(wasHandled);
#endif
}

bool RemoteScrollingCoordinatorProxyMac::scrollingTreeNodeRequestsScroll(ScrollingNodeID, const RequestedScrollData&)
{
    // Unlike iOS, we handle scrolling requests for the main frame in the same way we handle them for subscrollers.
    return false;
}

bool RemoteScrollingCoordinatorProxyMac::scrollingTreeNodeRequestsKeyboardScroll(ScrollingNodeID, const RequestedKeyboardScrollData&)
{
    // Unlike iOS, we handle scrolling requests for the main frame in the same way we handle them for subscrollers.
    return false;
}

void RemoteScrollingCoordinatorProxyMac::hasNodeWithAnimatedScrollChanged(bool hasAnimatedScrolls)
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->hasNodeWithAnimatedScrollChanged(hasAnimatedScrolls);
#else
    auto* drawingArea = dynamicDowncast<RemoteLayerTreeDrawingAreaProxy>(webPageProxy().drawingArea());
    if (!drawingArea)
        return;

    drawingArea->setDisplayLinkWantsFullSpeedUpdates(hasAnimatedScrolls);
#endif
}

void RemoteScrollingCoordinatorProxyMac::setRubberBandingInProgressForNode(ScrollingNodeID nodeID, bool isRubberBanding)
{
    if (isRubberBanding) {
        if (scrollingTree()->scrollingPerformanceTestingEnabled()) {
            m_eventDispatcher->didStartRubberbanding();
            webPageProxy().logScrollingEvent(static_cast<uint32_t>(PerformanceLoggingClient::ScrollingEvent::StartedRubberbanding), MonotonicTime::now(), 0);
        }
        m_uiState.addNodeWithActiveRubberband(nodeID);
    } else
        m_uiState.removeNodeWithActiveRubberband(nodeID);

    sendUIStateChangedIfNecessary();
}

void RemoteScrollingCoordinatorProxyMac::clearNodesWithUserScrollInProgress()
{
    m_uiState.clearNodesWithActiveUserScroll();
    sendUIStateChangedIfNecessary();
}

void RemoteScrollingCoordinatorProxyMac::scrollingTreeNodeWillStartScroll(ScrollingNodeID nodeID)
{
    m_uiState.addNodeWithActiveUserScroll(nodeID);
    sendUIStateChangedIfNecessary();
}

void RemoteScrollingCoordinatorProxyMac::scrollingTreeNodeDidEndScroll(ScrollingNodeID nodeID)
{
    m_uiState.removeNodeWithActiveUserScroll(nodeID);
    sendUIStateChangedIfNecessary();
}

void RemoteScrollingCoordinatorProxyMac::scrollingTreeNodeDidBeginScrollSnapping(ScrollingNodeID nodeID)
{
    m_uiState.addNodeWithActiveScrollSnap(nodeID);
    sendUIStateChangedIfNecessary();
}

void RemoteScrollingCoordinatorProxyMac::scrollingTreeNodeDidEndScrollSnapping(ScrollingNodeID nodeID)
{
    m_uiState.removeNodeWithActiveScrollSnap(nodeID);
    sendUIStateChangedIfNecessary();
}

void RemoteScrollingCoordinatorProxyMac::connectStateNodeLayers(ScrollingStateTree& stateTree, const RemoteLayerTreeHost& layerTreeHost)
{
    for (auto& currNode : stateTree.nodeMap().values()) {
        if (currNode->hasChangedProperty(ScrollingStateNode::Property::Layer))
            currNode->setLayer(layerTreeHost.layerForID(currNode->layer().layerID()));

        switch (currNode->nodeType()) {
        case ScrollingNodeType::MainFrame:
        case ScrollingNodeType::Subframe: {
            ScrollingStateFrameScrollingNode& scrollingStateNode = downcast<ScrollingStateFrameScrollingNode>(currNode);
            
            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ScrollContainerLayer))
                scrollingStateNode.setScrollContainerLayer(layerTreeHost.layerForID(scrollingStateNode.scrollContainerLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ScrolledContentsLayer))
                scrollingStateNode.setScrolledContentsLayer(layerTreeHost.layerForID(scrollingStateNode.scrolledContentsLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::CounterScrollingLayer))
                scrollingStateNode.setCounterScrollingLayer(layerTreeHost.layerForID(scrollingStateNode.counterScrollingLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::InsetClipLayer))
                scrollingStateNode.setInsetClipLayer(layerTreeHost.layerForID(scrollingStateNode.insetClipLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ContentShadowLayer))
                scrollingStateNode.setContentShadowLayer(layerTreeHost.layerForID(scrollingStateNode.contentShadowLayer().layerID()));

            // FIXME: we should never have header and footer layers coming from the WebProcess.
            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::HeaderLayer))
                scrollingStateNode.setHeaderLayer(layerTreeHost.layerForID(scrollingStateNode.headerLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::FooterLayer))
                scrollingStateNode.setFooterLayer(layerTreeHost.layerForID(scrollingStateNode.footerLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::VerticalScrollbarLayer))
                scrollingStateNode.setVerticalScrollbarLayer(layerTreeHost.layerForID(scrollingStateNode.verticalScrollbarLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::HorizontalScrollbarLayer))
                scrollingStateNode.setHorizontalScrollbarLayer(layerTreeHost.layerForID(scrollingStateNode.horizontalScrollbarLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::RootContentsLayer))
                scrollingStateNode.setRootContentsLayer(layerTreeHost.layerForID(scrollingStateNode.rootContentsLayer().layerID()));
            break;
        }
        case ScrollingNodeType::Overflow: {
            ScrollingStateOverflowScrollingNode& scrollingStateNode = downcast<ScrollingStateOverflowScrollingNode>(currNode);
            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ScrollContainerLayer))
                scrollingStateNode.setScrollContainerLayer(layerTreeHost.layerForID(scrollingStateNode.scrollContainerLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ScrolledContentsLayer))
                scrollingStateNode.setScrolledContentsLayer(layerTreeHost.layerForID(scrollingStateNode.scrolledContentsLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::VerticalScrollbarLayer))
                scrollingStateNode.setVerticalScrollbarLayer(layerTreeHost.layerForID(scrollingStateNode.verticalScrollbarLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::HorizontalScrollbarLayer))
                scrollingStateNode.setHorizontalScrollbarLayer(layerTreeHost.layerForID(scrollingStateNode.horizontalScrollbarLayer().layerID()));
            break;
        }
        case ScrollingNodeType::PluginScrolling: {
            ScrollingStatePluginScrollingNode& scrollingStateNode = downcast<ScrollingStatePluginScrollingNode>(currNode);
            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ScrollContainerLayer))
                scrollingStateNode.setScrollContainerLayer(layerTreeHost.layerForID(scrollingStateNode.scrollContainerLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ScrolledContentsLayer))
                scrollingStateNode.setScrolledContentsLayer(layerTreeHost.layerForID(scrollingStateNode.scrolledContentsLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::VerticalScrollbarLayer))
                scrollingStateNode.setVerticalScrollbarLayer(layerTreeHost.layerForID(scrollingStateNode.verticalScrollbarLayer().layerID()));

            if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::HorizontalScrollbarLayer))
                scrollingStateNode.setHorizontalScrollbarLayer(layerTreeHost.layerForID(scrollingStateNode.horizontalScrollbarLayer().layerID()));
            break;
        }
        case ScrollingNodeType::OverflowProxy:
        case ScrollingNodeType::FrameHosting:
        case ScrollingNodeType::PluginHosting:
        case ScrollingNodeType::Fixed:
        case ScrollingNodeType::Sticky:
        case ScrollingNodeType::Positioned:
            break;
        }
    }
}

void RemoteScrollingCoordinatorProxyMac::establishLayerTreeScrollingRelations(const RemoteLayerTreeHost&)
{
}

void RemoteScrollingCoordinatorProxyMac::displayDidRefresh(PlatformDisplayID displayID)
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->mainThreadDisplayDidRefresh(displayID);
#endif
}

void RemoteScrollingCoordinatorProxyMac::windowScreenDidChange(PlatformDisplayID displayID, std::optional<FramesPerSecond> nominalFramesPerSecond)
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->windowScreenDidChange(displayID, nominalFramesPerSecond);
#endif
}

void RemoteScrollingCoordinatorProxyMac::windowScreenWillChange()
{
#if ENABLE(SCROLLING_THREAD)
    m_eventDispatcher->windowScreenWillChange();
#endif
}

void RemoteScrollingCoordinatorProxyMac::willCommitLayerAndScrollingTrees()
{
    scrollingTree()->lockLayersForHitTesting();
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    m_eventDispatcher->lockForAnimationChanges();
#endif
}

void RemoteScrollingCoordinatorProxyMac::didCommitLayerAndScrollingTrees()
{
    scrollingTree()->unlockLayersForHitTesting();
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    m_eventDispatcher->unlockForAnimationChanges();
#endif
}

void RemoteScrollingCoordinatorProxyMac::applyScrollingTreeLayerPositionsAfterCommit()
{
    RemoteScrollingCoordinatorProxy::applyScrollingTreeLayerPositionsAfterCommit();
    m_eventDispatcher->renderingUpdateComplete();
}

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
void RemoteScrollingCoordinatorProxyMac::animationsWereAddedToNode(RemoteLayerTreeNode& node)
{
    m_eventDispatcher->animationsWereAddedToNode(node);
}

void RemoteScrollingCoordinatorProxyMac::animationsWereRemovedFromNode(RemoteLayerTreeNode& node)
{
    m_eventDispatcher->animationsWereRemovedFromNode(node);
}
#endif

} // namespace WebKit

#endif // PLATFORM(MAC) && ENABLE(UI_SIDE_COMPOSITING)
