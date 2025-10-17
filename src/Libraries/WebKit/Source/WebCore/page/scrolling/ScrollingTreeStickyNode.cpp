/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#include "ScrollingTreeStickyNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "Logging.h"
#include "ScrollingStateStickyNode.h"
#include "ScrollingTree.h"
#include "ScrollingTreeFixedNode.h"
#include "ScrollingTreeFrameScrollingNode.h"
#include "ScrollingTreeOverflowScrollProxyNode.h"
#include "ScrollingTreeOverflowScrollingNode.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeStickyNode);

ScrollingTreeStickyNode::ScrollingTreeStickyNode(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeNode(scrollingTree, ScrollingNodeType::Sticky, nodeID)
{
    scrollingTree.fixedOrStickyNodeAdded(*this);
}

ScrollingTreeStickyNode::~ScrollingTreeStickyNode() = default;

bool ScrollingTreeStickyNode::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    auto* stickyStateNode = dynamicDowncast<ScrollingStateStickyNode>(stateNode);
    if (!stickyStateNode)
        return false;

    if (stickyStateNode->hasChangedProperty(ScrollingStateNode::Property::ViewportConstraints))
        m_constraints = stickyStateNode->viewportConstraints();

    return true;
}

void ScrollingTreeStickyNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "sticky node";
    ScrollingTreeNode::dumpProperties(ts, behavior);
    ts.dumpProperty("sticky constraints", m_constraints);
    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeLayerPositions)
        ts.dumpProperty("layer top left", layerTopLeft());
}

FloatPoint ScrollingTreeStickyNode::computeLayerPosition() const
{
    FloatSize offsetFromStickyAncestors;
    auto computeLayerPositionForScrollingNode = [&](ScrollingTreeNode& scrollingNode) {
        FloatRect constrainingRect;
        if (auto* frameScrollingNode = dynamicDowncast<ScrollingTreeFrameScrollingNode>(scrollingNode))
            constrainingRect = frameScrollingNode->layoutViewport();
        else if (auto* overflowScrollingNode = dynamicDowncast<ScrollingTreeOverflowScrollingNode>(scrollingNode)) {
            constrainingRect = m_constraints.constrainingRectAtLastLayout();
            constrainingRect.move(overflowScrollingNode->scrollDeltaSinceLastCommit());
        }
        constrainingRect.move(-offsetFromStickyAncestors);
        return m_constraints.layerPositionForConstrainingRect(constrainingRect);
    };

    for (RefPtr ancestor = parent(); ancestor; ancestor = ancestor->parent()) {
        if (auto* overflowProxyNode = dynamicDowncast<ScrollingTreeOverflowScrollProxyNode>(*ancestor)) {
            auto overflowNode = scrollingTree()->nodeForID(overflowProxyNode->overflowScrollingNodeID());
            if (!overflowNode)
                break;

            return computeLayerPositionForScrollingNode(*overflowNode);
        }

        if (is<ScrollingTreeScrollingNode>(*ancestor))
            return computeLayerPositionForScrollingNode(*ancestor);

        if (auto* stickyNode = dynamicDowncast<ScrollingTreeStickyNode>(*ancestor))
            offsetFromStickyAncestors += stickyNode->scrollDeltaSinceLastCommit();

        if (is<ScrollingTreeFixedNode>(*ancestor)) {
            // FIXME: Do we need scrolling tree nodes at all for nested cases?
            return m_constraints.layerPositionAtLastLayout();
        }
    }
    ASSERT_NOT_REACHED();
    return m_constraints.layerPositionAtLastLayout();
}

FloatSize ScrollingTreeStickyNode::scrollDeltaSinceLastCommit() const
{
    auto layerPosition = computeLayerPosition();
    return layerPosition - m_constraints.layerPositionAtLastLayout();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
