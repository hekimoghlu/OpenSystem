/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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
#import "ScrollingTreePluginScrollingNodeIOS.h"

#if PLATFORM(IOS_FAMILY)
#if ENABLE(ASYNC_SCROLLING)

#import "ScrollingTreeScrollingNodeDelegateIOS.h"

#import <WebCore/ScrollingStatePluginScrollingNode.h>

namespace WebKit {
using namespace WebCore;

Ref<ScrollingTreePluginScrollingNodeIOS> ScrollingTreePluginScrollingNodeIOS::create(WebCore::ScrollingTree& scrollingTree, WebCore::ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreePluginScrollingNodeIOS(scrollingTree, nodeID));
}

ScrollingTreePluginScrollingNodeIOS::ScrollingTreePluginScrollingNodeIOS(WebCore::ScrollingTree& scrollingTree, WebCore::ScrollingNodeID nodeID)
    : ScrollingTreePluginScrollingNode(scrollingTree, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateIOS>(*this);
}

ScrollingTreePluginScrollingNodeIOS::~ScrollingTreePluginScrollingNodeIOS()
{
}

ScrollingTreeScrollingNodeDelegateIOS& ScrollingTreePluginScrollingNodeIOS::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateIOS*>(m_delegate.get());
}

WKBaseScrollView *ScrollingTreePluginScrollingNodeIOS::scrollView() const
{
    return delegate().scrollView();
}

bool ScrollingTreePluginScrollingNodeIOS::commitStateBeforeChildren(const WebCore::ScrollingStateNode& stateNode)
{
    if (!is<ScrollingStateScrollingNode>(stateNode))
        return false;

    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::ScrollContainerLayer))
        delegate().resetScrollViewDelegate();

    if (!ScrollingTreePluginScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    delegate().commitStateBeforeChildren(downcast<ScrollingStateScrollingNode>(stateNode));
    return true;
}

bool ScrollingTreePluginScrollingNodeIOS::commitStateAfterChildren(const ScrollingStateNode& stateNode)
{
    auto* scrollingStateNode = dynamicDowncast<ScrollingStateScrollingNode>(stateNode);
    if (!scrollingStateNode)
        return false;

    delegate().commitStateAfterChildren(*scrollingStateNode);

    return ScrollingTreePluginScrollingNode::commitStateAfterChildren(*scrollingStateNode);
}

void ScrollingTreePluginScrollingNodeIOS::repositionScrollingLayers()
{
    delegate().repositionScrollingLayers();
}

} // namespace WebKit

#endif // ENABLE(ASYNC_SCROLLING)
#endif // PLATFORM(IOS_FAMILY)
