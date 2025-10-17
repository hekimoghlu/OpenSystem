/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#include "RemoteScrollingTreeIOS.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(UI_SIDE_COMPOSITING)

#include "RemoteScrollingCoordinatorProxy.h"
#include "ScrollingTreeFrameScrollingNodeRemoteIOS.h"
#include "ScrollingTreeOverflowScrollingNodeIOS.h"
#include "ScrollingTreePluginScrollingNodeIOS.h"
#include <WebCore/ScrollingTreeFixedNodeCocoa.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteScrollingTreeIOS);

using namespace WebCore;

Ref<RemoteScrollingTree> RemoteScrollingTree::create(RemoteScrollingCoordinatorProxy& scrollingCoordinator)
{
    return adoptRef(*new RemoteScrollingTreeIOS(scrollingCoordinator));
}

RemoteScrollingTreeIOS::RemoteScrollingTreeIOS(RemoteScrollingCoordinatorProxy& scrollingCoordinatorProxy)
    : RemoteScrollingTree(scrollingCoordinatorProxy)
{
}

RemoteScrollingTreeIOS::~RemoteScrollingTreeIOS() = default;

void RemoteScrollingTreeIOS::scrollingTreeNodeDidScroll(ScrollingTreeScrollingNode& node, ScrollingLayerPositionAction scrollingLayerPositionAction)
{
    ASSERT(isMainRunLoop());

    // Scroll updates for the main frame on iOS are sent via WebPageProxy::updateVisibleContentRects()
    if (node.isRootNode()) {
        ScrollingTree::scrollingTreeNodeDidScroll(node, scrollingLayerPositionAction);
        return;
    }

    RemoteScrollingTree::scrollingTreeNodeDidScroll(node, scrollingLayerPositionAction);
}

void RemoteScrollingTreeIOS::scrollingTreeNodeWillStartPanGesture(ScrollingNodeID nodeID)
{
    if (m_scrollingCoordinatorProxy)
        m_scrollingCoordinatorProxy->scrollingTreeNodeWillStartPanGesture(nodeID);
}

Ref<ScrollingTreeNode> RemoteScrollingTreeIOS::createScrollingTreeNode(ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    switch (nodeType) {
    case ScrollingNodeType::MainFrame:
    case ScrollingNodeType::Subframe:
        return ScrollingTreeFrameScrollingNodeRemoteIOS::create(*this, nodeType, nodeID);

    case ScrollingNodeType::Overflow:
        return ScrollingTreeOverflowScrollingNodeIOS::create(*this, nodeID);

    case ScrollingNodeType::PluginScrolling:
        return ScrollingTreePluginScrollingNodeIOS::create(*this, nodeID);

    case ScrollingNodeType::FrameHosting:
    case ScrollingNodeType::PluginHosting:
    case ScrollingNodeType::OverflowProxy:
    case ScrollingNodeType::Fixed:
    case ScrollingNodeType::Sticky:
    case ScrollingNodeType::Positioned:
        return RemoteScrollingTree::createScrollingTreeNode(nodeType, nodeID);
    }
    ASSERT_NOT_REACHED();
    return ScrollingTreeFixedNodeCocoa::create(*this, nodeID);
}

} // namespace WebKit

#endif // #if PLATFORM(IOS_FAMILY) && ENABLE(UI_SIDE_COMPOSITING)
