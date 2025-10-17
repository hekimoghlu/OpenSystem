/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

#include "MessageReceiver.h"
#include "RemoteScrollingCoordinator.h"
#include "RemoteScrollingTree.h"
#include "RemoteScrollingUIState.h"
#include <WebCore/PlatformLayer.h>
#include <WebCore/PlatformLayerIdentifier.h>
#include <WebCore/ScrollSnapOffsetsInfo.h>
#include <WebCore/WheelEventTestMonitor.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS UIScrollView;

namespace WebCore {
class FloatPoint;
class PlatformWheelEvent;
}

namespace WebKit {

class NativeWebWheelEvent;
class RemoteLayerTreeHost;
class RemoteLayerTreeNode;
class RemoteScrollingCoordinatorTransaction;
class RemoteScrollingTree;
class WebPageProxy;
class WebWheelEvent;

class RemoteScrollingCoordinatorProxy : public CanMakeWeakPtr<RemoteScrollingCoordinatorProxy>, public CanMakeCheckedPtr<RemoteScrollingCoordinatorProxy> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingCoordinatorProxy);
    WTF_MAKE_NONCOPYABLE(RemoteScrollingCoordinatorProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteScrollingCoordinatorProxy);
public:
    virtual ~RemoteScrollingCoordinatorProxy();
    
    constexpr bool isRemoteScrollingCoordinatorProxyIOS() const
    {
#if PLATFORM(IOS_FAMILY)
        return true;
#else
        return false;
#endif
    }

    constexpr bool isRemoteScrollingCoordinatorProxyMac() const
    {
#if PLATFORM(MAC)
        return true;
#else
        return false;
#endif
    }

    // Inform the web process that the scroll position changed (called from the scrolling tree)
    virtual bool scrollingTreeNodeRequestsScroll(WebCore::ScrollingNodeID, const WebCore::RequestedScrollData&);
    virtual bool scrollingTreeNodeRequestsKeyboardScroll(WebCore::ScrollingNodeID, const WebCore::RequestedKeyboardScrollData&);
    void scrollingTreeNodeDidStopAnimatedScroll(WebCore::ScrollingNodeID);

    void scrollingThreadAddedPendingUpdate();

    WebCore::TrackingType eventTrackingTypeForPoint(WebCore::EventTrackingRegions::EventType, WebCore::IntPoint) const;

    // Called externally when native views move around.
    void viewportChangedViaDelegatedScrolling(const WebCore::FloatPoint& scrollPosition, const WebCore::FloatRect& layoutViewport, double scale);

    virtual void applyScrollingTreeLayerPositionsAfterCommit();

    void currentSnapPointIndicesDidChange(WebCore::ScrollingNodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical);

    virtual void cacheWheelEventScrollingAccelerationCurve(const NativeWebWheelEvent&) { }
    virtual void handleWheelEvent(const WebWheelEvent&, WebCore::RectEdges<bool> rubberBandableEdges);
    void continueWheelEventHandling(const WebWheelEvent&, WebCore::WheelEventHandlingResult);
    virtual void wheelEventHandlingCompleted(const WebCore::PlatformWheelEvent&, std::optional<WebCore::ScrollingNodeID>, std::optional<WebCore::WheelScrollGestureState>, bool /* wasHandled */) { }

    virtual WebCore::PlatformWheelEvent filteredWheelEvent(const WebCore::PlatformWheelEvent& wheelEvent) { return wheelEvent; }

    std::optional<WebCore::ScrollingNodeID> rootScrollingNodeID() const;

    const RemoteLayerTreeHost* layerTreeHost() const;
    WebPageProxy& webPageProxy() const;
    Ref<WebPageProxy> protectedWebPageProxy() const;

    std::optional<WebCore::RequestedScrollData> commitScrollingTreeState(IPC::Connection&, const RemoteScrollingCoordinatorTransaction&, std::optional<WebCore::LayerHostingContextIdentifier> = std::nullopt);

    bool hasFixedOrSticky() const;
    bool hasScrollableMainFrame() const;
    bool hasScrollableOrZoomedMainFrame() const;

    WebCore::ScrollbarWidth mainFrameScrollbarWidth() const;

    WebCore::OverscrollBehavior mainFrameHorizontalOverscrollBehavior() const;
    WebCore::OverscrollBehavior mainFrameVerticalOverscrollBehavior() const;

    virtual void scrollingTreeNodeWillStartPanGesture(WebCore::ScrollingNodeID) { }
    virtual void scrollingTreeNodeWillStartScroll(WebCore::ScrollingNodeID) { }
    virtual void scrollingTreeNodeDidEndScroll(WebCore::ScrollingNodeID) { }
    virtual void clearNodesWithUserScrollInProgress() { }
    virtual void hasNodeWithAnimatedScrollChanged(bool) { }
    virtual void setRootNodeIsInUserScroll(bool) { }
    virtual void setRubberBandingInProgressForNode(WebCore::ScrollingNodeID, bool isRubberBanding) { }

    virtual void scrollingTreeNodeDidBeginScrollSnapping(WebCore::ScrollingNodeID) { }
    virtual void scrollingTreeNodeDidEndScrollSnapping(WebCore::ScrollingNodeID) { }
    
    virtual void willCommitLayerAndScrollingTrees() { }
    virtual void didCommitLayerAndScrollingTrees() { }

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    virtual void animationsWereAddedToNode(RemoteLayerTreeNode&) { }
    virtual void animationsWereRemovedFromNode(RemoteLayerTreeNode&) { }
#endif

    String scrollingTreeAsText() const;

    void resetStateAfterProcessExited();

    virtual void displayDidRefresh(WebCore::PlatformDisplayID);
    void reportExposedUnfilledArea(MonotonicTime, unsigned unfilledArea);
    void reportSynchronousScrollingReasonsChanged(MonotonicTime, OptionSet<WebCore::SynchronousScrollingReason>);
    void reportFilledVisibleFreshTile(MonotonicTime, unsigned);
    bool scrollingPerformanceTestingEnabled() const;
    
    void receivedWheelEventWithPhases(WebCore::PlatformWheelEventPhase phase, WebCore::PlatformWheelEventPhase momentumPhase);
    void deferWheelEventTestCompletionForReason(std::optional<WebCore::ScrollingNodeID>, WebCore::WheelEventTestMonitor::DeferReason);
    void removeWheelEventTestCompletionDeferralForReason(std::optional<WebCore::ScrollingNodeID>, WebCore::WheelEventTestMonitor::DeferReason);

    virtual void windowScreenWillChange() { }
    virtual void windowScreenDidChange(WebCore::PlatformDisplayID, std::optional<WebCore::FramesPerSecond>) { }

    float topContentInset() const;
    WebCore::FloatPoint currentMainFrameScrollPosition() const;
    WebCore::FloatRect computeVisibleContentRect();
    WebCore::IntPoint scrollOrigin() const;
    int headerHeight() const;
    int footerHeight() const;
    float mainFrameScaleFactor() const;
    WebCore::FloatSize totalContentsSize() const;
    
    void viewWillStartLiveResize();
    void viewWillEndLiveResize();
    void viewSizeDidChange();
    String scrollbarStateForScrollingNodeID(std::optional<WebCore::ScrollingNodeID>, bool isVertical);
    bool overlayScrollbarsEnabled();

    void sendScrollingTreeNodeDidScroll();
    
    void scrollingTreeNodeScrollbarVisibilityDidChange(WebCore::ScrollingNodeID, WebCore::ScrollbarOrientation, bool);
    void scrollingTreeNodeScrollbarMinimumThumbLengthDidChange(WebCore::ScrollingNodeID, WebCore::ScrollbarOrientation, int);
    void receivedLastScrollingTreeNodeDidScrollReply();
    bool isMonitoringWheelEvents();

protected:
    explicit RemoteScrollingCoordinatorProxy(WebPageProxy&);

    RemoteScrollingTree* scrollingTree() const { return m_scrollingTree.get(); }

    virtual void connectStateNodeLayers(WebCore::ScrollingStateTree&, const RemoteLayerTreeHost&) = 0;
    virtual void establishLayerTreeScrollingRelations(const RemoteLayerTreeHost&) = 0;

    virtual void didReceiveWheelEvent(bool /* wasHandled */) { }

    void sendUIStateChangedIfNecessary();

private:
    WeakRef<WebPageProxy> m_webPageProxy;
    RefPtr<RemoteScrollingTree> m_scrollingTree;

protected:
    std::optional<WebCore::RequestedScrollData> m_requestedScroll;
    RemoteScrollingUIState m_uiState;
    std::optional<unsigned> m_currentHorizontalSnapPointIndex;
    std::optional<unsigned> m_currentVerticalSnapPointIndex;
    bool m_waitingForDidScrollReply { false };
    HashSet<WebCore::PlatformLayerIdentifier> m_layersWithScrollingRelations;
};

} // namespace WebKit

#define SPECIALIZE_TYPE_TRAITS_REMOTE_SCROLLING_COORDINATOR_PROXY(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::ToValueTypeName) \
    static bool isType(const WebKit::RemoteScrollingCoordinatorProxy& scrollingCoordinatorProxy) { return scrollingCoordinatorProxy.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(UI_SIDE_COMPOSITING)
