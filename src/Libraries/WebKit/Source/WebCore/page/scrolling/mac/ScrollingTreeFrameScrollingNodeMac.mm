/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#import "ScrollingTreeFrameScrollingNodeMac.h"

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#import "LayoutSize.h"
#import "LocalFrameView.h"
#import "Logging.h"
#import "PlatformWheelEvent.h"
#import "ScrollableArea.h"
#import "ScrollingCoordinator.h"
#import "ScrollingStateTree.h"
#import "ScrollingThread.h"
#import "ScrollingTree.h"
#import "ScrollingTreeScrollingNodeDelegateMac.h"
#import "TileController.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/BlockObjCExceptions.h>
#import <wtf/Deque.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/CString.h>
#import <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeFrameScrollingNodeMac);

Ref<ScrollingTreeFrameScrollingNode> ScrollingTreeFrameScrollingNodeMac::create(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFrameScrollingNodeMac(scrollingTree, nodeType, nodeID));
}

ScrollingTreeFrameScrollingNodeMac::ScrollingTreeFrameScrollingNodeMac(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
    : ScrollingTreeFrameScrollingNode(scrollingTree, nodeType, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateMac>(*this);
}

ScrollingTreeFrameScrollingNodeMac::~ScrollingTreeFrameScrollingNodeMac() = default;

ScrollingTreeScrollingNodeDelegateMac& ScrollingTreeFrameScrollingNodeMac::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateMac*>(m_delegate.get());
}

void ScrollingTreeFrameScrollingNodeMac::willBeDestroyed()
{
    delegate().nodeWillBeDestroyed();
}

bool ScrollingTreeFrameScrollingNodeMac::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreeFrameScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    auto* scrollingStateNode = dynamicDowncast<ScrollingStateFrameScrollingNode>(stateNode);
    if (!scrollingStateNode)
        return false;

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::RootContentsLayer))
        m_rootContentsLayer = static_cast<CALayer*>(scrollingStateNode->rootContentsLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::CounterScrollingLayer))
        m_counterScrollingLayer = static_cast<CALayer*>(scrollingStateNode->counterScrollingLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::InsetClipLayer))
        m_insetClipLayer = static_cast<CALayer*>(scrollingStateNode->insetClipLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::ContentShadowLayer))
        m_contentShadowLayer = static_cast<CALayer*>(scrollingStateNode->contentShadowLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::HeaderLayer))
        m_headerLayer = static_cast<CALayer*>(scrollingStateNode->headerLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::FooterLayer))
        m_footerLayer = static_cast<CALayer*>(scrollingStateNode->footerLayer());

    bool logScrollingMode = !m_hadFirstUpdate;
    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::ReasonsForSynchronousScrolling))
        logScrollingMode = true;

    if (logScrollingMode && isRootNode() && scrollingTree()->scrollingPerformanceTestingEnabled())
        scrollingTree()->reportSynchronousScrollingReasonsChanged(MonotonicTime::now(), synchronousScrollingReasons());

    m_delegate->updateFromStateNode(*scrollingStateNode);

    m_hadFirstUpdate = true;
    return true;
}

bool ScrollingTreeFrameScrollingNodeMac::commitStateAfterChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreeFrameScrollingNode::commitStateAfterChildren(stateNode))
        return false;

    auto* scrollingStateNode = dynamicDowncast<ScrollingStateScrollingNode>(stateNode);
    if (!scrollingStateNode)
        return false;

    if (isRootNode()
        && (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::ScrolledContentsLayer)
        || scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::TotalContentsSize)
        || scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::ScrollableAreaSize)))
        updateMainFramePinAndRubberbandState();

    return true;
}

WheelEventHandlingResult ScrollingTreeFrameScrollingNodeMac::handleWheelEvent(const PlatformWheelEvent& wheelEvent, EventTargeting eventTargeting)
{
    if (!canHandleWheelEvent(wheelEvent, eventTargeting))
        return WheelEventHandlingResult::unhandled();

    bool handled = delegate().handleWheelEvent(wheelEvent);
    delegate().updateSnapScrollState();
    return WheelEventHandlingResult::result(handled);
}

void ScrollingTreeFrameScrollingNodeMac::willDoProgrammaticScroll(const FloatPoint& targetScrollPosition)
{
    delegate().willDoProgrammaticScroll(targetScrollPosition);
}

void ScrollingTreeFrameScrollingNodeMac::currentScrollPositionChanged(ScrollType scrollType, ScrollingLayerPositionAction action)
{
    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeFrameScrollingNodeMac " << scrollingNodeID() << " currentScrollPositionChanged to " << currentScrollPosition() << " min: " << minimumScrollPosition() << " max: " << maximumScrollPosition() << " sync: " << hasSynchronousScrollingReasons() << " is animating: " << scrollingTree()->isScrollAnimationInProgressForNode(scrollingNodeID()));

    delegate().currentScrollPositionChanged();

    if (isRootNode())
        updateMainFramePinAndRubberbandState();

    ScrollingTreeFrameScrollingNode::currentScrollPositionChanged(scrollType, hasSynchronousScrollingReasons() ? ScrollingLayerPositionAction::Set : action);

    if (scrollingTree()->scrollingPerformanceTestingEnabled()) {
        unsigned unfilledArea = exposedUnfilledArea();
        if (unfilledArea || m_lastScrollHadUnfilledPixels)
            scrollingTree()->reportExposedUnfilledArea(MonotonicTime::now(), unfilledArea);

        m_lastScrollHadUnfilledPixels = unfilledArea;
    }
}

void ScrollingTreeFrameScrollingNodeMac::repositionScrollingLayers()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    auto* layer = static_cast<CALayer*>(scrolledContentsLayer());
    if (ScrollingThread::isCurrentThread()) {
        // If we're committing on the scrolling thread, it means that ThreadedScrollingTree is in "desynchronized" mode.
        // The main thread may already have set the same layer position, but here we need to trigger a scrolling thread commit to
        // ensure that the scroll happens even when the main thread commit is taking a long time. So make sure the layer property changes
        // when there has been a scroll position change.
        if (!scrollingTree()->isScrollingSynchronizedWithMainThread())
            layer.position = CGPointZero;
    }

    // We use scroll position here because the root content layer is offset to account for scrollOrigin (see LocalFrameView::positionForRootContentLayer).
    layer.position = -currentScrollPosition();
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollingTreeFrameScrollingNodeMac::repositionRelatedLayers()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    auto scrollPosition = currentScrollPosition();
    auto layoutViewport = this->layoutViewport();

    FloatRect visibleContentRect(scrollPosition, scrollableAreaSize());

    if (m_counterScrollingLayer)
        m_counterScrollingLayer.get().position = layoutViewport.location();

    float topContentInset = this->topContentInset();
    if (m_insetClipLayer && m_rootContentsLayer) {
        m_insetClipLayer.get().position = FloatPoint(m_insetClipLayer.get().position.x, LocalFrameView::yPositionForInsetClipLayer(scrollPosition, topContentInset));
        m_rootContentsLayer.get().position = LocalFrameView::positionForRootContentLayer(scrollPosition, scrollOrigin(), topContentInset, headerHeight());
        if (m_contentShadowLayer)
            m_contentShadowLayer.get().position = m_rootContentsLayer.get().position;
    }

    if (m_headerLayer || m_footerLayer) {
        // Generally the banners should have the same horizontal-position computation as a fixed element. However,
        // the banners are not affected by the frameScaleFactor(), so if there is currently a non-1 frameScaleFactor()
        // then we should recompute layoutViewport.x() for the banner with a scale factor of 1.
        float horizontalScrollOffsetForBanner = layoutViewport.x();
        if (m_headerLayer)
            m_headerLayer.get().position = FloatPoint(horizontalScrollOffsetForBanner, LocalFrameView::yPositionForHeaderLayer(scrollPosition, topContentInset));

        if (m_footerLayer)
            m_footerLayer.get().position = FloatPoint(horizontalScrollOffsetForBanner, LocalFrameView::yPositionForFooterLayer(scrollPosition, topContentInset, totalContentsSize().height(), footerHeight()));
    }
    END_BLOCK_OBJC_EXCEPTIONS

    delegate().updateScrollbarPainters();
}

FloatPoint ScrollingTreeFrameScrollingNodeMac::minimumScrollPosition() const
{
    FloatPoint position = ScrollableArea::scrollPositionFromOffset(FloatPoint(), toFloatSize(scrollOrigin()));
    
    if (isRootNode() && scrollingTree()->scrollPinningBehavior() == ScrollPinningBehavior::PinToBottom)
        position.setY(maximumScrollPosition().y());

    return position;
}

FloatPoint ScrollingTreeFrameScrollingNodeMac::maximumScrollPosition() const
{
    FloatPoint position = ScrollableArea::scrollPositionFromOffset(FloatPoint(totalContentsSizeForRubberBand() - scrollableAreaSize()), toFloatSize(scrollOrigin()));
    position = position.expandedTo(FloatPoint());

    if (isRootNode() && scrollingTree()->scrollPinningBehavior() == ScrollPinningBehavior::PinToTop)
        position.setY(minimumScrollPosition().y());

    return position;
}

void ScrollingTreeFrameScrollingNodeMac::updateMainFramePinAndRubberbandState()
{
    ASSERT(isRootNode());
    scrollingTree()->setMainFramePinnedState(edgePinnedState());
}

unsigned ScrollingTreeFrameScrollingNodeMac::exposedUnfilledArea() const
{
    Region paintedVisibleTiles;

    Deque<CALayer*> layerQueue;
    layerQueue.append(scrolledContentsLayer());
    PlatformLayerList tiles;

    while (!layerQueue.isEmpty() && tiles.isEmpty()) {
        CALayer* layer = layerQueue.takeFirst();
        auto sublayers = adoptNS([[layer sublayers] copy]);

        // If this layer is the parent of a tile, it is the parent of all of the tiles and nothing else.
        if ([[[sublayers objectAtIndex:0] valueForKey:@"isTile"] boolValue]) {
            for (CALayer* sublayer in sublayers.get())
                tiles.append(sublayer);
        } else {
            for (CALayer* sublayer in sublayers.get())
                layerQueue.append(sublayer);
        }
    }

    FloatPoint clampedScrollPosition = clampScrollPosition(currentScrollPosition());
    FloatRect viewportRect({ }, scrollableAreaSize());
    return TileController::blankPixelCountForTiles(tiles, viewportRect, IntPoint(-clampedScrollPosition.x(), -clampedScrollPosition.y()));
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)
