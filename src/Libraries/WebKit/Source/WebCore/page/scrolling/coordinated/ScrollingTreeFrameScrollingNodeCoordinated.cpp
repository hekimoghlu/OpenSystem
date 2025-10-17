/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#include "ScrollingTreeFrameScrollingNodeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "LocalFrameView.h"
#include "Logging.h"
#include "ScrollingStateFrameScrollingNode.h"
#include "ScrollingThread.h"
#include "ScrollingTreeScrollingNodeDelegateCoordinated.h"
#include "ThreadedScrollingTree.h"

namespace WebCore {

Ref<ScrollingTreeFrameScrollingNode> ScrollingTreeFrameScrollingNodeCoordinated::create(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFrameScrollingNodeCoordinated(scrollingTree, nodeType, nodeID));
}

ScrollingTreeFrameScrollingNodeCoordinated::ScrollingTreeFrameScrollingNodeCoordinated(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
    : ScrollingTreeFrameScrollingNode(scrollingTree, nodeType, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateCoordinated>(*this, downcast<ThreadedScrollingTree>(scrollingTree).scrollAnimatorEnabled());
}

ScrollingTreeFrameScrollingNodeCoordinated::~ScrollingTreeFrameScrollingNodeCoordinated() = default;

ScrollingTreeScrollingNodeDelegateCoordinated& ScrollingTreeFrameScrollingNodeCoordinated::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateCoordinated*>(m_delegate.get());
}

bool ScrollingTreeFrameScrollingNodeCoordinated::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreeFrameScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    if (!is<ScrollingStateFrameScrollingNode>(stateNode))
        return false;

    const auto& scrollingStateNode = downcast<ScrollingStateFrameScrollingNode>(stateNode);

    if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::RootContentsLayer))
        m_rootContentsLayer = static_cast<CoordinatedPlatformLayer*>(scrollingStateNode.rootContentsLayer());
    if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::CounterScrollingLayer))
        m_counterScrollingLayer = static_cast<CoordinatedPlatformLayer*>(scrollingStateNode.counterScrollingLayer());
    if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::InsetClipLayer))
        m_insetClipLayer = static_cast<CoordinatedPlatformLayer*>(scrollingStateNode.insetClipLayer());
    if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::ContentShadowLayer))
        m_contentShadowLayer = static_cast<CoordinatedPlatformLayer*>(scrollingStateNode.contentShadowLayer());
    if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::HeaderLayer))
        m_headerLayer = static_cast<CoordinatedPlatformLayer*>(scrollingStateNode.headerLayer());
    if (scrollingStateNode.hasChangedProperty(ScrollingStateNode::Property::FooterLayer))
        m_footerLayer = static_cast<CoordinatedPlatformLayer*>(scrollingStateNode.footerLayer());

    m_delegate->updateFromStateNode(scrollingStateNode);
    return true;
}

WheelEventHandlingResult ScrollingTreeFrameScrollingNodeCoordinated::handleWheelEvent(const PlatformWheelEvent& wheelEvent, EventTargeting eventTargeting)
{
    if (!canHandleWheelEvent(wheelEvent, eventTargeting))
        return WheelEventHandlingResult::unhandled();

    bool handled = delegate().handleWheelEvent(wheelEvent);
    delegate().updateSnapScrollState();
    return WheelEventHandlingResult::result(handled);
}

void ScrollingTreeFrameScrollingNodeCoordinated::currentScrollPositionChanged(ScrollType scrollType, ScrollingLayerPositionAction action)
{
    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeFrameScrollingNodeCoordinated::currentScrollPositionChanged to " << currentScrollPosition() << " min: " << minimumScrollPosition() << " max: " << maximumScrollPosition() << " sync: " << hasSynchronousScrollingReasons());

    ScrollingTreeFrameScrollingNode::currentScrollPositionChanged(scrollType, hasSynchronousScrollingReasons() ? ScrollingLayerPositionAction::Set : action);
}

void ScrollingTreeFrameScrollingNodeCoordinated::repositionScrollingLayers()
{
    auto* scrollLayer = static_cast<CoordinatedPlatformLayer*>(scrolledContentsLayer());
    if (!scrollLayer)
        return;

    // If we're committing on the scrolling thread, it means that ThreadedScrollingTree is in "desynchronized" mode.
    // The main thread may already have set the same layer position, but here we need to trigger a scrolling thread composition
    // to ensure that the scroll happens even when the main thread commit is taking a long time. So make sure the layer property
    // changes when there has been a scroll position change.
    CoordinatedPlatformLayer::ForcePositionSync forceSync = ScrollingThread::isCurrentThread() && !scrollingTree()->isScrollingSynchronizedWithMainThread() ?
        CoordinatedPlatformLayer::ForcePositionSync::Yes : CoordinatedPlatformLayer::ForcePositionSync::No;

    auto scrollPosition = currentScrollPosition();
    scrollLayer->setPositionForScrolling(-scrollPosition, forceSync);
}

void ScrollingTreeFrameScrollingNodeCoordinated::repositionRelatedLayers()
{
    auto scrollPosition = currentScrollPosition();
    auto layoutViewport = this->layoutViewport();

    if (m_counterScrollingLayer)
        m_counterScrollingLayer->setPositionForScrolling(layoutViewport.location());

    float topContentInset = this->topContentInset();
    if (m_insetClipLayer && m_rootContentsLayer) {
        FloatPoint insetClipPosition;
        {
            Locker locker { m_insetClipLayer->lock() };
            insetClipPosition = FloatPoint(m_insetClipLayer->position().x(), LocalFrameView::yPositionForInsetClipLayer(scrollPosition, topContentInset));
        }
        m_insetClipLayer->setPositionForScrolling(insetClipPosition);
        auto rootContentsPosition = LocalFrameView::positionForRootContentLayer(scrollPosition, scrollOrigin(), topContentInset, headerHeight());
        m_rootContentsLayer->setPositionForScrolling(rootContentsPosition);
        if (m_contentShadowLayer)
            m_contentShadowLayer->setPositionForScrolling(rootContentsPosition);
    }

    if (m_headerLayer || m_footerLayer) {
        // Generally the banners should have the same horizontal-position computation as a fixed element. However,
        // the banners are not affected by the frameScaleFactor(), so if there is currently a non-1 frameScaleFactor()
        // then we should recompute layoutViewport.x() for the banner with a scale factor of 1.
        float horizontalScrollOffsetForBanner = layoutViewport.x();
        if (m_headerLayer)
            m_headerLayer->setPositionForScrolling(FloatPoint(horizontalScrollOffsetForBanner, LocalFrameView::yPositionForHeaderLayer(scrollPosition, topContentInset)));
        if (m_footerLayer)
            m_footerLayer->setPositionForScrolling(FloatPoint(horizontalScrollOffsetForBanner, LocalFrameView::yPositionForFooterLayer(scrollPosition, topContentInset, totalContentsSize().height(), footerHeight())));
    }

    delegate().updateVisibleLengths();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
