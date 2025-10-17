/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#import "ScrollingTreeOverflowScrollingNodeIOS.h"

#if PLATFORM(IOS_FAMILY)
#if ENABLE(ASYNC_SCROLLING)

#import "ScrollingTreeScrollingNodeDelegateIOS.h"

#import <WebCore/ScrollingStateOverflowScrollingNode.h>

namespace WebKit {
using namespace WebCore;

Ref<ScrollingTreeOverflowScrollingNodeIOS> ScrollingTreeOverflowScrollingNodeIOS::create(WebCore::ScrollingTree& scrollingTree, WebCore::ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeOverflowScrollingNodeIOS(scrollingTree, nodeID));
}

ScrollingTreeOverflowScrollingNodeIOS::ScrollingTreeOverflowScrollingNodeIOS(WebCore::ScrollingTree& scrollingTree, WebCore::ScrollingNodeID nodeID)
    : ScrollingTreeOverflowScrollingNode(scrollingTree, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateIOS>(*this);
}

ScrollingTreeOverflowScrollingNodeIOS::~ScrollingTreeOverflowScrollingNodeIOS()
{
}

ScrollingTreeScrollingNodeDelegateIOS& ScrollingTreeOverflowScrollingNodeIOS::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateIOS*>(m_delegate.get());
}

WKBaseScrollView *ScrollingTreeOverflowScrollingNodeIOS::scrollView() const
{
    return delegate().scrollView();
}

bool ScrollingTreeOverflowScrollingNodeIOS::commitStateBeforeChildren(const WebCore::ScrollingStateNode& stateNode)
{
    if (!is<ScrollingStateScrollingNode>(stateNode))
        return false;

    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::ScrollContainerLayer))
        delegate().resetScrollViewDelegate();

    if (!ScrollingTreeOverflowScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    delegate().commitStateBeforeChildren(downcast<ScrollingStateScrollingNode>(stateNode));
    return true;
}

bool ScrollingTreeOverflowScrollingNodeIOS::commitStateAfterChildren(const ScrollingStateNode& stateNode)
{
    auto* scrollingStateNode = dynamicDowncast<ScrollingStateScrollingNode>(stateNode);
    if (!scrollingStateNode)
        return false;

    delegate().commitStateAfterChildren(*scrollingStateNode);

    return ScrollingTreeOverflowScrollingNode::commitStateAfterChildren(*scrollingStateNode);
}

void ScrollingTreeOverflowScrollingNodeIOS::repositionScrollingLayers()
{
    delegate().repositionScrollingLayers();
}

String ScrollingTreeOverflowScrollingNodeIOS::scrollbarStateForOrientation(ScrollbarOrientation orientation) const
{
    if (auto* scrollView = this->scrollView()) {
        auto showsScrollbar = orientation == ScrollbarOrientation::Horizontal ?  [scrollView showsHorizontalScrollIndicator] :  [scrollView showsVerticalScrollIndicator];
        return showsScrollbar ? ""_s : "none"_s;
    }
    return ""_s;
}

} // namespace WebKit

#endif // ENABLE(ASYNC_SCROLLING)
#endif // PLATFORM(IOS_FAMILY)
