/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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

#if PLATFORM(IOS_FAMILY) && ENABLE(ASYNC_SCROLLING)

#include "RemoteScrollingCoordinatorProxy.h"
#include <wtf/TZoneMalloc.h>

OBJC_CLASS UIScrollView;
OBJC_CLASS WKBaseScrollView;

namespace WebCore {
enum class TouchAction : uint8_t;
}

namespace WebKit {

class RemoteLayerTreeDrawingAreaProxyIOS;
class RemoteLayerTreeNode;

class RemoteScrollingCoordinatorProxyIOS final : public RemoteScrollingCoordinatorProxy {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingCoordinatorProxyIOS);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteScrollingCoordinatorProxyIOS);
public:
    explicit RemoteScrollingCoordinatorProxyIOS(WebPageProxy&);
    ~RemoteScrollingCoordinatorProxyIOS() = default;

    UIScrollView *scrollViewForScrollingNodeID(std::optional<WebCore::ScrollingNodeID>) const;

    OptionSet<WebCore::TouchAction> activeTouchActionsForTouchIdentifier(unsigned touchIdentifier) const;
    void setTouchActionsForTouchIdentifier(OptionSet<WebCore::TouchAction>, unsigned);
    void clearTouchActionsForTouchIdentifier(unsigned);

    bool shouldSetScrollViewDecelerationRateFast() const;
    void setRootNodeIsInUserScroll(bool) override;

    void adjustTargetContentOffsetForSnapping(CGSize maxScrollDimensions, CGPoint velocity, CGFloat topInset, CGPoint currentContentOffset, CGPoint* targetContentOffset);
    bool hasActiveSnapPoint() const;
    CGPoint nearestActiveContentInsetAdjustedSnapOffset(CGFloat topInset, const CGPoint&) const;

#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
    void removeDestroyedLayerIDs(const Vector<WebCore::PlatformLayerIdentifier>&);
    void updateOverlayRegionLayerIDs(const HashSet<WebCore::PlatformLayerIdentifier>& overlayRegionLayerIDs) { m_overlayRegionLayerIDs = overlayRegionLayerIDs; }

    const HashSet<WebCore::PlatformLayerIdentifier>& fixedScrollingNodeLayerIDs() const { return m_fixedScrollingNodeLayerIDs; }
    const HashSet<WebCore::PlatformLayerIdentifier>& overlayRegionLayerIDs() const { return m_overlayRegionLayerIDs; }
    Vector<WKBaseScrollView*> overlayRegionScrollViewCandidates() const;
#endif

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    void animationsWereAddedToNode(RemoteLayerTreeNode&) override WTF_IGNORES_THREAD_SAFETY_ANALYSIS;
    void animationsWereRemovedFromNode(RemoteLayerTreeNode&) override;
    void updateAnimations();
#endif

    void displayDidRefresh(WebCore::PlatformDisplayID) override;

private:
    RemoteLayerTreeDrawingAreaProxyIOS& drawingAreaIOS() const;

    void scrollingTreeNodeWillStartPanGesture(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeWillStartScroll(WebCore::ScrollingNodeID) override;
    void scrollingTreeNodeDidEndScroll(WebCore::ScrollingNodeID) override;

    void connectStateNodeLayers(WebCore::ScrollingStateTree&, const RemoteLayerTreeHost&) override;
    void establishLayerTreeScrollingRelations(const RemoteLayerTreeHost&) override;

    WebCore::FloatRect currentLayoutViewport() const;

    bool shouldSnapForMainFrameScrolling(WebCore::ScrollEventAxis) const;
    std::pair<float, std::optional<unsigned>> closestSnapOffsetForMainFrameScrolling(WebCore::ScrollEventAxis, float currentScrollOffset, WebCore::FloatPoint scrollDestination, float velocity) const;

    HashMap<unsigned, OptionSet<WebCore::TouchAction>> m_touchActionsByTouchIdentifier;

#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
    HashSet<WebCore::PlatformLayerIdentifier> m_fixedScrollingNodeLayerIDs;
    HashSet<WebCore::PlatformLayerIdentifier> m_overlayRegionLayerIDs;
    HashMap<WebCore::PlatformLayerIdentifier, WebCore::ScrollingNodeID> m_scrollingNodesByLayerID;
#endif

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    HashSet<WebCore::PlatformLayerIdentifier> m_animatedNodeLayerIDs;
#endif
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_REMOTE_SCROLLING_COORDINATOR_PROXY(RemoteScrollingCoordinatorProxyIOS, isRemoteScrollingCoordinatorProxyIOS());

#endif // PLATFORM(IOS_FAMILY) && ENABLE(ASYNC_SCROLLING)
