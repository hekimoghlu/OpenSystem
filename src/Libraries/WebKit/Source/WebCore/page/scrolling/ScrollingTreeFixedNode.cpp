/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "ScrollingTreeFixedNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateFixedNode.h"
#include "ScrollingThread.h"
#include "ScrollingTree.h"
#include "ScrollingTreeFrameScrollingNode.h"
#include "ScrollingTreeOverflowScrollProxyNode.h"
#include "ScrollingTreeOverflowScrollingNode.h"
#include "ScrollingTreePositionedNode.h"
#include "ScrollingTreeStickyNode.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeFixedNode);

ScrollingTreeFixedNode::ScrollingTreeFixedNode(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeNode(scrollingTree, ScrollingNodeType::Fixed, nodeID)
{
    scrollingTree.fixedOrStickyNodeAdded(*this);
}

ScrollingTreeFixedNode::~ScrollingTreeFixedNode() = default;

bool ScrollingTreeFixedNode::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    auto* fixedStateNode = dynamicDowncast<ScrollingStateFixedNode>(stateNode);
    if (!fixedStateNode)
        return false;

    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::ViewportConstraints))
        m_constraints = fixedStateNode->viewportConstraints();

    return true;
}

FloatPoint ScrollingTreeFixedNode::computeLayerPosition() const
{
    FloatSize overflowScrollDelta;
    ScrollingTreeStickyNode* lastStickyNode = nullptr;
    for (RefPtr ancestor = parent(); ancestor; ancestor = ancestor->parent()) {
        if (auto* scrollingNode = dynamicDowncast<ScrollingTreeFrameScrollingNode>(*ancestor)) {
            // Fixed nodes are positioned relative to the containing frame scrolling node.
            // We bail out after finding one.
            auto layoutViewport = scrollingNode->layoutViewport();
            return m_constraints.layerPositionForViewportRect(layoutViewport) - overflowScrollDelta;
        }

        if (auto* overflowNode = dynamicDowncast<ScrollingTreeOverflowScrollingNode>(*ancestor)) {
            // To keep the layer still during async scrolling we adjust by how much the position has changed since layout.
            overflowScrollDelta -= overflowNode->scrollDeltaSinceLastCommit();
            continue;
        }

        if (auto* overflowNode = dynamicDowncast<ScrollingTreeOverflowScrollProxyNode>(*ancestor)) {
            // To keep the layer still during async scrolling we adjust by how much the position has changed since layout.
            overflowScrollDelta -= overflowNode->scrollDeltaSinceLastCommit();
            continue;
        }

        if (auto* positioningAncestor = dynamicDowncast<ScrollingTreePositionedNode>(*ancestor)) {
            // See if sticky node already handled this positioning node.
            // FIXME: Include positioning node information to sticky/fixed node to avoid these tests.
            if (lastStickyNode && lastStickyNode->layer() == positioningAncestor->layer())
                continue;
            if (positioningAncestor->layer() != layer())
                overflowScrollDelta -= positioningAncestor->scrollDeltaSinceLastCommit();
            continue;
        }

        if (auto* stickyNode = dynamicDowncast<ScrollingTreeStickyNode>(*ancestor)) {
            overflowScrollDelta += stickyNode->scrollDeltaSinceLastCommit();
            lastStickyNode = stickyNode;
            continue;
        }

        if (is<ScrollingTreeFixedNode>(*ancestor)) {
            // The ancestor fixed node has already applied the needed corrections to say put.
            return m_constraints.layerPositionAtLastLayout() - overflowScrollDelta;
        }
    }
    ASSERT_NOT_REACHED();
    return FloatPoint();
}

void ScrollingTreeFixedNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "fixed node";
    ScrollingTreeNode::dumpProperties(ts, behavior);
    ts.dumpProperty("fixed constraints", m_constraints);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
