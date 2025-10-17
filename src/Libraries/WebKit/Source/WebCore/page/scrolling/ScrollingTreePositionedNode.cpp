/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include "ScrollingTreePositionedNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "Logging.h"
#include "ScrollingStatePositionedNode.h"
#include "ScrollingTree.h"
#include "ScrollingTreeOverflowScrollingNode.h"
#include "ScrollingTreeScrollingNode.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreePositionedNode);

ScrollingTreePositionedNode::ScrollingTreePositionedNode(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeNode(scrollingTree, ScrollingNodeType::Positioned, nodeID)
{
}

ScrollingTreePositionedNode::~ScrollingTreePositionedNode() = default;

bool ScrollingTreePositionedNode::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    auto* positionedStateNode = dynamicDowncast<ScrollingStatePositionedNode>(stateNode);
    if (!positionedStateNode)
        return false;

    if (positionedStateNode->hasChangedProperty(ScrollingStateNode::Property::RelatedOverflowScrollingNodes))
        m_relatedOverflowScrollingNodes = positionedStateNode->relatedOverflowScrollingNodes();

    if (positionedStateNode->hasChangedProperty(ScrollingStateNode::Property::LayoutConstraintData))
        m_constraints = positionedStateNode->layoutConstraints();

    if (!m_relatedOverflowScrollingNodes.isEmpty())
        scrollingTree()->activePositionedNodes().add(*this);

    return true;
}

FloatSize ScrollingTreePositionedNode::scrollDeltaSinceLastCommit() const
{
    FloatSize delta;
    for (auto nodeID : m_relatedOverflowScrollingNodes) {
        if (auto* node = dynamicDowncast<ScrollingTreeOverflowScrollingNode>(scrollingTree()->nodeForID(nodeID)))
            delta += node->scrollDeltaSinceLastCommit();
    }

    // Positioned nodes compensate for scrolling, so negate the scroll delta.
    return -delta;
}

void ScrollingTreePositionedNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "positioned node";
    ScrollingTreeNode::dumpProperties(ts, behavior);

    ts.dumpProperty("layout constraints", m_constraints);
    ts.dumpProperty("related overflow nodes", m_relatedOverflowScrollingNodes.size());

    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeNodeIDs) {
        if (!m_relatedOverflowScrollingNodes.isEmpty()) {
            TextStream::GroupScope scope(ts);
            ts << "overflow nodes";
            for (auto nodeID : m_relatedOverflowScrollingNodes)
                ts << "\n" << indent << "nodeID " << nodeID;
        }
    }
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
