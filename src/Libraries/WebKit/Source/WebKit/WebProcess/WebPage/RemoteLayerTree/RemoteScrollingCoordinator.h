/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#if ENABLE(ASYNC_SCROLLING)

#include "MessageReceiver.h"
#include <WebCore/AsyncScrollingCoordinator.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class WebPage;
class RemoteScrollingCoordinatorTransaction;
class RemoteScrollingUIState;

class RemoteScrollingCoordinator final : public WebCore::AsyncScrollingCoordinator, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingCoordinator);
public:
    static Ref<RemoteScrollingCoordinator> create(WebPage* page)
    {
        return adoptRef(*new RemoteScrollingCoordinator(page));
    }

    void ref() const final { WebCore::AsyncScrollingCoordinator::ref(); }
    void deref() const final { WebCore::AsyncScrollingCoordinator::deref(); }

    RemoteScrollingCoordinatorTransaction buildTransaction(WebCore::FrameIdentifier);

    void scrollingStateInUIProcessChanged(const RemoteScrollingUIState&);

    void addNodeWithActiveRubberBanding(WebCore::ScrollingNodeID);
    void removeNodeWithActiveRubberBanding(WebCore::ScrollingNodeID);
    
    void setCurrentWheelEventWillStartSwipe(std::optional<bool> value) { m_currentWheelEventWillStartSwipe = value; }

    struct NodeAndGestureState {
        std::optional<WebCore::ScrollingNodeID> wheelGestureNode;
        std::optional<WebCore::WheelScrollGestureState> wheelGestureState;
    };

    NodeAndGestureState takeCurrentWheelGestureInfo() { return std::exchange(m_currentWheelGestureInfo, { }); }

private:
    RemoteScrollingCoordinator(WebPage*);
    virtual ~RemoteScrollingCoordinator();

    bool isRemoteScrollingCoordinator() const override { return true; }
    
    // ScrollingCoordinator
    bool coordinatesScrollingForFrameView(const WebCore::LocalFrameView&) const override;
    void scheduleTreeStateCommit() override;

    bool isRubberBandInProgress(std::optional<WebCore::ScrollingNodeID>) const final;
    bool isUserScrollInProgress(std::optional<WebCore::ScrollingNodeID>) const final;
    bool isScrollSnapInProgress(std::optional<WebCore::ScrollingNodeID>) const final;

    void setScrollPinningBehavior(WebCore::ScrollPinningBehavior) override;
    
    void startMonitoringWheelEvents(bool clearLatchingState) final;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    
    // Respond to UI process changes.
    void scrollPositionChangedForNode(WebCore::ScrollingNodeID, const WebCore::FloatPoint& scrollPosition, std::optional<WebCore::FloatPoint> layoutViewportOrigin, bool syncLayerPosition, CompletionHandler<void()>&&);
    void animatedScrollDidEndForNode(WebCore::ScrollingNodeID);
    void currentSnapPointIndicesChangedForNode(WebCore::ScrollingNodeID, std::optional<unsigned> horizontal, std::optional<unsigned> vertical);

    void receivedWheelEventWithPhases(WebCore::PlatformWheelEventPhase phase, WebCore::PlatformWheelEventPhase momentumPhase);
    void startDeferringScrollingTestCompletionForNode(WebCore::ScrollingNodeID, OptionSet<WebCore::WheelEventTestMonitor::DeferReason>);
    void stopDeferringScrollingTestCompletionForNode(WebCore::ScrollingNodeID, OptionSet<WebCore::WheelEventTestMonitor::DeferReason>);
    void scrollingTreeNodeScrollbarVisibilityDidChange(WebCore::ScrollingNodeID, WebCore::ScrollbarOrientation, bool);
    void scrollingTreeNodeScrollbarMinimumThumbLengthDidChange(WebCore::ScrollingNodeID nodeID, WebCore::ScrollbarOrientation orientation, int minimumThumbLength);

    WebCore::WheelEventHandlingResult handleWheelEventForScrolling(const WebCore::PlatformWheelEvent&, WebCore::ScrollingNodeID, std::optional<WebCore::WheelScrollGestureState>) override;

    WeakPtr<WebPage> m_webPage;

    HashSet<WebCore::ScrollingNodeID> m_nodesWithActiveRubberBanding;
    HashSet<WebCore::ScrollingNodeID> m_nodesWithActiveScrollSnap;
    HashSet<WebCore::ScrollingNodeID> m_nodesWithActiveUserScrolls;

    NodeAndGestureState m_currentWheelGestureInfo;

    bool m_clearScrollLatchingInNextTransaction { false };
    
    std::optional<bool> m_currentWheelEventWillStartSwipe;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_SCROLLING_COORDINATOR(WebKit::RemoteScrollingCoordinator, isRemoteScrollingCoordinator());

#endif // ENABLE(ASYNC_SCROLLING)
