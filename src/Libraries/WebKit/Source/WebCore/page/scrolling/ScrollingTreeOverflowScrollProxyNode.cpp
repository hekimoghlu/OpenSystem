/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#include "ScrollingTreeOverflowScrollProxyNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateOverflowScrollProxyNode.h"
#include "ScrollingStateTree.h"
#include "ScrollingTree.h"
#include "ScrollingTreeOverflowScrollingNode.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeOverflowScrollProxyNode);

ScrollingTreeOverflowScrollProxyNode::ScrollingTreeOverflowScrollProxyNode(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeNode(scrollingTree, ScrollingNodeType::OverflowProxy, nodeID)
{
}

ScrollingTreeOverflowScrollProxyNode::~ScrollingTreeOverflowScrollProxyNode() = default;

bool ScrollingTreeOverflowScrollProxyNode::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    auto* overflowProxyStateNode = dynamicDowncast<ScrollingStateOverflowScrollProxyNode>(stateNode);
    if (!overflowProxyStateNode)
        return false;

    if (overflowProxyStateNode->hasChangedProperty(ScrollingStateNode::Property::OverflowScrollingNode))
        m_overflowScrollingNodeID = overflowProxyStateNode->overflowScrollingNode();

    if (m_overflowScrollingNodeID) {
        auto& relatedNodes = scrollingTree()->overflowRelatedNodes();
        relatedNodes.ensure(*m_overflowScrollingNodeID, [] {
            return Vector<ScrollingNodeID>();
        }).iterator->value.append(scrollingNodeID());

        scrollingTree()->activeOverflowScrollProxyNodes().add(*this);
    }
    
    return true;
}

FloatSize ScrollingTreeOverflowScrollProxyNode::scrollDeltaSinceLastCommit() const
{
    if (auto* node = dynamicDowncast<ScrollingTreeOverflowScrollingNode>(scrollingTree()->nodeForID(m_overflowScrollingNodeID)))
        return node->scrollDeltaSinceLastCommit();

    return { };
}

FloatPoint ScrollingTreeOverflowScrollProxyNode::computeLayerPosition() const
{
    FloatPoint scrollOffset;
    if (auto* node = dynamicDowncast<ScrollingTreeOverflowScrollingNode>(scrollingTree()->nodeForID(m_overflowScrollingNodeID)))
        scrollOffset = node->currentScrollOffset();
    return scrollOffset;
}

void ScrollingTreeOverflowScrollProxyNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "overflow scroll proxy node";
    ScrollingTreeNode::dumpProperties(ts, behavior);

    if (auto* relatedOverflowNode = scrollingTree()->nodeForID(m_overflowScrollingNodeID)) {
        if (RefPtr scrollingNode = dynamicDowncast<ScrollingTreeOverflowScrollingNode>(relatedOverflowNode))
            ts.dumpProperty("related overflow scrolling node scroll position", scrollingNode->currentScrollPosition());
    }

    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeNodeIDs)
        ts.dumpProperty("overflow scrolling node", m_overflowScrollingNodeID);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
