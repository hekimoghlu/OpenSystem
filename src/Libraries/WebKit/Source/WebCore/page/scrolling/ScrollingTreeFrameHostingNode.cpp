/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#include "ScrollingTreeFrameHostingNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "Logging.h"
#include "ScrollingStateFrameHostingNode.h"
#include "ScrollingStateTree.h"
#include "ScrollingTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeFrameHostingNode);

Ref<ScrollingTreeFrameHostingNode> ScrollingTreeFrameHostingNode::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFrameHostingNode(scrollingTree, nodeID));
}

ScrollingTreeFrameHostingNode::ScrollingTreeFrameHostingNode(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeNode(scrollingTree, ScrollingNodeType::FrameHosting, nodeID)
{
    ASSERT(isFrameHostingNode());
}

ScrollingTreeFrameHostingNode::~ScrollingTreeFrameHostingNode() = default;

bool ScrollingTreeFrameHostingNode::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    auto* state = dynamicDowncast<ScrollingStateFrameHostingNode>(stateNode);
    if (!state)
        return false;

    if (state->hasChangedProperty(ScrollingStateNode::Property::LayerHostingContextIdentifier))
        setLayerHostingContextIdentifier(state->layerHostingContextIdentifier());
    return true;
}

void ScrollingTreeFrameHostingNode::setLayerHostingContextIdentifier(std::optional<LayerHostingContextIdentifier> identifier)
{
    if (m_hostingContext != identifier)
        removeHostedChildren();
    m_hostingContext = identifier;
    if (m_hostingContext)
        scrollingTree()->addScrollingNodeToHostedSubtreeMap(*m_hostingContext, *this);
}

void ScrollingTreeFrameHostingNode::removeHostedChildren()
{
    auto hostedChildren = std::exchange(m_hostedChildren, { });
    for (auto& children : hostedChildren)
        scrollingTree()->removeNode(children->scrollingNodeID());
}

void ScrollingTreeFrameHostingNode::willBeDestroyed()
{
    if (m_hostingContext)
        scrollingTree()->removeFrameHostingNode(*m_hostingContext);
    removeHostedChildren();
}

void ScrollingTreeFrameHostingNode::removeHostedChild(RefPtr<ScrollingTreeNode> node)
{
    if (node) {
        m_hostedChildren.remove(node);
        m_children.removeFirst(node.releaseNonNull());
    }
}

void ScrollingTreeFrameHostingNode::applyLayerPositions()
{
}

void ScrollingTreeFrameHostingNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "frame hosting node";
    if (auto hostingContextIdentifier = m_hostingContext) {
        if (behavior & ScrollingStateTreeAsTextBehavior::IncludeNodeIDs)
            ts.dumpProperty("hosting context identifier", *m_hostingContext);
        else
            ts.dumpProperty("has hosting context identifier", "");
    }
    ScrollingTreeNode::dumpProperties(ts, behavior);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
