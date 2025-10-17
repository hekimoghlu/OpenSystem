/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#import "ScrollingTreePositionedNodeCocoa.h"

#if ENABLE(ASYNC_SCROLLING)

#import "Logging.h"
#import "ScrollingStatePositionedNode.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreePositionedNodeCocoa);

Ref<ScrollingTreePositionedNodeCocoa> ScrollingTreePositionedNodeCocoa::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreePositionedNodeCocoa(scrollingTree, nodeID));
}

ScrollingTreePositionedNodeCocoa::ScrollingTreePositionedNodeCocoa(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreePositionedNode(scrollingTree, nodeID)
{
}

ScrollingTreePositionedNodeCocoa::~ScrollingTreePositionedNodeCocoa() = default;

bool ScrollingTreePositionedNodeCocoa::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CALayer*>(stateNode.layer());

    return ScrollingTreePositionedNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreePositionedNodeCocoa::applyLayerPositions()
{
    auto delta = scrollDeltaSinceLastCommit();
    auto layerPosition = m_constraints.layerPositionAtLastLayout() - delta;

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreePositionedNode " << scrollingNodeID() << " applyLayerPositions: overflow delta " << delta << " moving layer to " << layerPosition);

    [m_layer _web_setLayerTopLeftPosition:layerPosition - m_constraints.alignmentOffset()];
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
