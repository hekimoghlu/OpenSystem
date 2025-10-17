/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include "ScrollingTreeNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateTree.h"
#include "ScrollingTree.h"
#include "ScrollingTreeFrameScrollingNode.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeNode);

ScrollingTreeNode::ScrollingTreeNode(ScrollingTree& scrollingTree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
    : m_scrollingTree(&scrollingTree)
    , m_nodeType(nodeType)
    , m_nodeID(nodeID)
{
}

ScrollingTreeNode::~ScrollingTreeNode() = default;

void ScrollingTreeNode::appendChild(Ref<ScrollingTreeNode>&& childNode)
{
    RELEASE_ASSERT(scrollingTree()->inCommitTreeState());

    childNode->setParent(this);

    m_children.append(WTFMove(childNode));
}

void ScrollingTreeNode::removeChild(ScrollingTreeNode& node)
{
    RELEASE_ASSERT(scrollingTree()->inCommitTreeState());

    size_t index = m_children.findIf([&](auto& child) {
        return &node == child.ptr();
    });

    // The index will be notFound if the node to remove is a deeper-than-1-level descendant or
    // if node is the root state node.
    if (index != notFound) {
        m_children.remove(index);
        return;
    }

    for (auto& child : m_children)
        child->removeChild(node);
}

void ScrollingTreeNode::removeAllChildren()
{
    RELEASE_ASSERT(scrollingTree()->inCommitTreeState());

    m_children.clear();
}

bool ScrollingTreeNode::isRootNode() const
{
    return scrollingTree()->rootNode() == this;
}

void ScrollingTreeNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeNodeIDs)
        ts.dumpProperty("nodeID", scrollingNodeID());
}

RefPtr<ScrollingTreeFrameScrollingNode> ScrollingTreeNode::enclosingFrameNodeIncludingSelf()
{
    RefPtr node = this;
    while (node && !node->isFrameScrollingNode())
        node = node->parent();

    return downcast<ScrollingTreeFrameScrollingNode>(node.get());
}

RefPtr<ScrollingTreeScrollingNode> ScrollingTreeNode::enclosingScrollingNodeIncludingSelf()
{
    RefPtr node = this;
    while (node && !node->isScrollingNode())
        node = node->parent();

    return downcast<ScrollingTreeScrollingNode>(node.get());
}

void ScrollingTreeNode::dump(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    dumpProperties(ts, behavior);

    for (auto& child : m_children) {
        TextStream::GroupScope scope(ts);
        child->dump(ts, behavior);
    }
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
