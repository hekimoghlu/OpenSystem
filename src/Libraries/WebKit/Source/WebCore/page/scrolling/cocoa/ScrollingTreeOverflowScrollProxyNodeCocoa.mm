/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#import "ScrollingTreeOverflowScrollProxyNodeCocoa.h"

#if ENABLE(ASYNC_SCROLLING)

#import "ScrollingStateOverflowScrollProxyNode.h"
#import "ScrollingStateTree.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeOverflowScrollProxyNodeCocoa);

Ref<ScrollingTreeOverflowScrollProxyNodeCocoa> ScrollingTreeOverflowScrollProxyNodeCocoa::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeOverflowScrollProxyNodeCocoa(scrollingTree, nodeID));
}

ScrollingTreeOverflowScrollProxyNodeCocoa::ScrollingTreeOverflowScrollProxyNodeCocoa(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeOverflowScrollProxyNode(scrollingTree, nodeID)
{
}

ScrollingTreeOverflowScrollProxyNodeCocoa::~ScrollingTreeOverflowScrollProxyNodeCocoa() = default;

bool ScrollingTreeOverflowScrollProxyNodeCocoa::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CALayer*>(stateNode.layer());

    return ScrollingTreeOverflowScrollProxyNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreeOverflowScrollProxyNodeCocoa::applyLayerPositions()
{
    FloatPoint scrollOffset = computeLayerPosition();

    LOG_WITH_STREAM(ScrollingTree, stream << "ScrollingTreeOverflowScrollProxyNodeCocoa " << scrollingNodeID() << " applyLayerPositions: setting bounds origin to " << scrollOffset);
    [m_layer _web_setLayerBoundsOrigin:scrollOffset];
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
