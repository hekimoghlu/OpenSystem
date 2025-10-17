/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#import "ScrollingTreeFrameScrollingNodeRemoteIOS.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(ASYNC_SCROLLING)

#import "ScrollingTreeScrollingNodeDelegateIOS.h"
#import <WebCore/ScrollingStateFrameScrollingNode.h>
#import <WebCore/ScrollingStateScrollingNode.h>
#import <WebCore/ScrollingTree.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeFrameScrollingNodeRemoteIOS);

using namespace WebCore;

Ref<ScrollingTreeFrameScrollingNodeRemoteIOS> ScrollingTreeFrameScrollingNodeRemoteIOS::create(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFrameScrollingNodeRemoteIOS(scrollingTree, nodeType, nodeID));
}

ScrollingTreeFrameScrollingNodeRemoteIOS::ScrollingTreeFrameScrollingNodeRemoteIOS(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
    : ScrollingTreeFrameScrollingNode(scrollingTree, nodeType, nodeID)
{
}

ScrollingTreeFrameScrollingNodeRemoteIOS::~ScrollingTreeFrameScrollingNodeRemoteIOS()
{
}

ScrollingTreeScrollingNodeDelegateIOS* ScrollingTreeFrameScrollingNodeRemoteIOS::delegate() const
{
    return static_cast<ScrollingTreeScrollingNodeDelegateIOS*>(m_delegate.get());
}

WKBaseScrollView *ScrollingTreeFrameScrollingNodeRemoteIOS::scrollView() const
{
    return m_delegate ? delegate()->scrollView() : nil;
}

bool ScrollingTreeFrameScrollingNodeRemoteIOS::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreeFrameScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    auto* scrollingStateNode = dynamicDowncast<ScrollingStateFrameScrollingNode>(stateNode);
    if (!scrollingStateNode)
        return false;

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::CounterScrollingLayer))
        m_counterScrollingLayer = static_cast<CALayer*>(scrollingStateNode->counterScrollingLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::HeaderLayer))
        m_headerLayer = static_cast<CALayer*>(scrollingStateNode->headerLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::FooterLayer))
        m_footerLayer = static_cast<CALayer*>(scrollingStateNode->footerLayer());

    if (scrollingStateNode->hasChangedProperty(ScrollingStateNode::Property::ScrollContainerLayer)) {
        if (scrollContainerLayer())
            m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateIOS>(*this);
        else
            m_delegate = nullptr;
    }

    if (m_delegate)
        delegate()->commitStateBeforeChildren(*scrollingStateNode);

    return true;
}

bool ScrollingTreeFrameScrollingNodeRemoteIOS::commitStateAfterChildren(const ScrollingStateNode& stateNode)
{
    if (m_delegate) {
        auto* scrollingStateNode = dynamicDowncast<ScrollingStateFrameScrollingNode>(stateNode);
        if (!scrollingStateNode)
            return false;

        delegate()->commitStateAfterChildren(*scrollingStateNode);
    }

    return ScrollingTreeFrameScrollingNode::commitStateAfterChildren(stateNode);
}

FloatPoint ScrollingTreeFrameScrollingNodeRemoteIOS::minimumScrollPosition() const
{
    FloatPoint position = ScrollableArea::scrollPositionFromOffset(FloatPoint(), toFloatSize(scrollOrigin()));
    
    if (isRootNode() && scrollingTree()->scrollPinningBehavior() == ScrollPinningBehavior::PinToBottom)
        position.setY(maximumScrollPosition().y());

    return position;
}

FloatPoint ScrollingTreeFrameScrollingNodeRemoteIOS::maximumScrollPosition() const
{
    FloatPoint position = ScrollableArea::scrollPositionFromOffset(FloatPoint(totalContentsSizeForRubberBand() - scrollableAreaSize()), toFloatSize(scrollOrigin()));
    position = position.expandedTo(FloatPoint());

    if (isRootNode() && scrollingTree()->scrollPinningBehavior() == ScrollPinningBehavior::PinToTop)
        position.setY(minimumScrollPosition().y());

    return position;
}

void ScrollingTreeFrameScrollingNodeRemoteIOS::repositionScrollingLayers()
{
    if (m_delegate) {
        delegate()->repositionScrollingLayers();
        return;
    }

    // Main frame scrolling is handled by the main UIScrollView.
}

void ScrollingTreeFrameScrollingNodeRemoteIOS::repositionRelatedLayers()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    auto layoutViewport = this->layoutViewport();

    [m_counterScrollingLayer setPosition:layoutViewport.location()];

    // FIXME: I don' think we never have headers and footers on iOS.
    if (m_headerLayer || m_footerLayer) {
        // Generally the banners should have the same horizontal-position computation as a fixed element. However,
        // the banners are not affected by the frameScaleFactor(), so if there is currently a non-1 frameScaleFactor()
        // then we should recompute scrollPositionForFixedChildren for the banner with a scale factor of 1.
        if (m_headerLayer)
            [m_headerLayer setPosition:FloatPoint(layoutViewport.x(), 0)];

        if (m_footerLayer)
            [m_footerLayer setPosition:FloatPoint(layoutViewport.x(), totalContentsSize().height() - footerHeight())];
    }
    END_BLOCK_OBJC_EXCEPTIONS
}

String ScrollingTreeFrameScrollingNodeRemoteIOS::scrollbarStateForOrientation(ScrollbarOrientation orientation) const
{
    if (auto* scrollView = this->scrollView()) {
        auto showsScrollbar = orientation == ScrollbarOrientation::Horizontal ?  [scrollView showsHorizontalScrollIndicator] :  [scrollView showsVerticalScrollIndicator];
        return showsScrollbar ? ""_s : "none"_s;
    }
    return ""_s;
}


}
#endif
