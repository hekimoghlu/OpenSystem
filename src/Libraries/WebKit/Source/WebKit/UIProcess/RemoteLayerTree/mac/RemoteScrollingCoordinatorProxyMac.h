/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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

#if PLATFORM(MAC) && ENABLE(UI_SIDE_COMPOSITING)

#include "RemoteLayerTreeEventDispatcher.h"
#include "RemoteScrollingCoordinatorProxy.h"
#include <wtf/TZoneMalloc.h>

namespace WebKit {

#if ENABLE(SCROLLING_THREAD)
class RemoteLayerTreeEventDispatcher;
#endif

class RemoteScrollingCoordinatorProxyMac final : public RemoteScrollingCoordinatorProxy {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingCoordinatorProxyMac);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteScrollingCoordinatorProxyMac);
public:
    explicit RemoteScrollingCoordinatorProxyMac(WebPageProxy&);
    ~RemoteScrollingCoordinatorProxyMac();

private:
    void cacheWheelEventScrollingAccelerationCurve(const NativeWebWheelEvent&) override;

    void handleWheelEvent(const WebWheelEvent&, WebCore::RectEdges<bool> rubberBandableEdges) override;
    void wheelEventHandlingCompleted(const WebCore::PlatformWheelEvent&, std::optional<WebCore::ScrollingNodeID>, std::optional<WebCore::WheelScrollGestureState>, bool wasHandled) override;

    bool scrollingTreeNodeRequestsScroll(WebCore::ScrollingNodeID, const WebCore::RequestedScrollData&) override;
    bool scrollingTreeNodeRequestsKeyboardScroll(WebCore::ScrollingNodeID, const WebCore::RequestedKeyboardScrollData&) override;
    void hasNodeWithAnimatedScrollChanged(bool) override;
    void setRubberBandingInProgressForNode(WebCore::ScrollingNodeID, bool isRubberBanding) override;

    void scrollingTreeNodeWillStartScroll(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScroll(WebCore::ScrollingNodeID) override;
    void clearNodesWithUserScrollInProgress() override;

    void scrollingTreeNodeDidBeginScrollSnapping(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScrollSnapping(WebCore::ScrollingNodeID) override;

    void connectStateNodeLayers(WebCore::ScrollingStateTree&, const RemoteLayerTreeHost&) override;
    void establishLayerTreeScrollingRelations(const RemoteLayerTreeHost&) override;

    void displayDidRefresh(WebCore::PlatformDisplayID) override;
    void windowScreenWillChange() override;
    void windowScreenDidChange(WebCore::PlatformDisplayID, std::optional<WebCore::FramesPerSecond>) override;

    void applyScrollingTreeLayerPositionsAfterCommit() override;

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    void willCommitLayerAndScrollingTrees() override WTF_ACQUIRES_LOCK(m_eventDispatcher->m_effectStacksLock);
    void didCommitLayerAndScrollingTrees() override WTF_RELEASES_LOCK(m_eventDispatcher->m_effectStacksLock);

    void animationsWereAddedToNode(RemoteLayerTreeNode&) override;
    void animationsWereRemovedFromNode(RemoteLayerTreeNode&) override;
#else
    void willCommitLayerAndScrollingTrees() override;
    void didCommitLayerAndScrollingTrees() override;
#endif

#if ENABLE(SCROLLING_THREAD)
    RefPtr<RemoteLayerTreeEventDispatcher> m_eventDispatcher;
#endif
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_REMOTE_SCROLLING_COORDINATOR_PROXY(RemoteScrollingCoordinatorProxyMac, isRemoteScrollingCoordinatorProxyMac());

#endif // PLATFORM(MAC) && ENABLE(UI_SIDE_COMPOSITING)
