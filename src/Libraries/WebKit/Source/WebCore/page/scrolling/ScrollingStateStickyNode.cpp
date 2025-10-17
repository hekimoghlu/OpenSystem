/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include "ScrollingStateStickyNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "GraphicsLayer.h"
#include "Logging.h"
#include "ScrollingStateFixedNode.h"
#include "ScrollingStateFrameScrollingNode.h"
#include "ScrollingStateOverflowScrollProxyNode.h"
#include "ScrollingStateOverflowScrollingNode.h"
#include "ScrollingStateTree.h"
#include "ScrollingTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingStateStickyNode);

ScrollingStateStickyNode::ScrollingStateStickyNode(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID, StickyPositionViewportConstraints&& constraints)
    : ScrollingStateNode(ScrollingNodeType::Sticky, nodeID, WTFMove(children), changedProperties, layerID)
    , m_constraints(WTFMove(constraints))
{
}

ScrollingStateStickyNode::ScrollingStateStickyNode(ScrollingStateTree& tree, ScrollingNodeID nodeID)
    : ScrollingStateNode(ScrollingNodeType::Sticky, tree, nodeID)
{
}

ScrollingStateStickyNode::ScrollingStateStickyNode(const ScrollingStateStickyNode& node, ScrollingStateTree& adoptiveTree)
    : ScrollingStateNode(node, adoptiveTree)
    , m_constraints(StickyPositionViewportConstraints(node.viewportConstraints()))
{
}

ScrollingStateStickyNode::~ScrollingStateStickyNode() = default;

Ref<ScrollingStateNode> ScrollingStateStickyNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStateStickyNode(*this, adoptiveTree));
}

OptionSet<ScrollingStateNode::Property> ScrollingStateStickyNode::applicableProperties() const
{
    constexpr OptionSet<Property> nodeProperties = { Property::ViewportConstraints };

    auto properties = ScrollingStateNode::applicableProperties();
    properties.add(nodeProperties);
    return properties;
}

void ScrollingStateStickyNode::updateConstraints(const StickyPositionViewportConstraints& constraints)
{
    if (m_constraints == constraints)
        return;

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingStateStickyNode " << scrollingNodeID() << " updateConstraints with constraining rect " << constraints.constrainingRectAtLastLayout() << " sticky offset " << constraints.stickyOffsetAtLastLayout() << " layer pos at last layout " << constraints.layerPositionAtLastLayout());

    m_constraints = constraints;
    setPropertyChanged(Property::ViewportConstraints);
}

FloatPoint ScrollingStateStickyNode::computeLayerPosition(const LayoutRect& viewportRect) const
{
    // This logic follows ScrollingTreeStickyNode::computeLayerPosition().
    FloatSize offsetFromStickyAncestors;
    auto computeLayerPositionForScrollingNode = [&](ScrollingStateNode& scrollingStateNode) {
        FloatRect constrainingRect;
        if (is<ScrollingStateFrameScrollingNode>(scrollingStateNode))
            constrainingRect = viewportRect;
        else if (RefPtr overflowScrollingNode = dynamicDowncast<ScrollingStateOverflowScrollingNode>(scrollingStateNode))
            constrainingRect = FloatRect(overflowScrollingNode->scrollPosition(), m_constraints.constrainingRectAtLastLayout().size());

        constrainingRect.move(offsetFromStickyAncestors);
        return m_constraints.layerPositionForConstrainingRect(constrainingRect);
    };

    for (auto ancestor = parent(); ancestor; ancestor = ancestor->parent()) {
        if (auto* overflowProxyNode = dynamicDowncast<ScrollingStateOverflowScrollProxyNode>(*ancestor)) {
            auto overflowNode = scrollingStateTree().stateNodeForID(overflowProxyNode->overflowScrollingNode());
            if (!overflowNode)
                break;

            return computeLayerPositionForScrollingNode(*overflowNode);
        }

        if (is<ScrollingStateScrollingNode>(*ancestor))
            return computeLayerPositionForScrollingNode(*ancestor);

        if (auto* stickyNode = dynamicDowncast<ScrollingStateStickyNode>(*ancestor))
            offsetFromStickyAncestors += stickyNode->scrollDeltaSinceLastCommit(viewportRect);

        if (is<ScrollingStateFixedNode>(*ancestor)) {
            // FIXME: Do we need scrolling tree nodes at all for nested cases?
            return m_constraints.layerPositionAtLastLayout();
        }
    }
    ASSERT_NOT_REACHED();
    return m_constraints.layerPositionAtLastLayout();
}

void ScrollingStateStickyNode::reconcileLayerPositionForViewportRect(const LayoutRect& viewportRect, ScrollingLayerPositionAction action)
{
    FloatPoint position = computeLayerPosition(viewportRect);
    if (layer().representsGraphicsLayer()) {
        auto* graphicsLayer = static_cast<GraphicsLayer*>(layer());
        ASSERT(graphicsLayer);
        // Crash data suggest that graphicsLayer can be null: rdar://106547410.
        if (!graphicsLayer)
            return;

        LOG_WITH_STREAM(Compositing, stream << "ScrollingStateStickyNode " << scrollingNodeID() << " reconcileLayerPositionForViewportRect " << action << " position of layer " << graphicsLayer->primaryLayerID() << " to " << position << " sticky offset " << m_constraints.stickyOffsetAtLastLayout());
        
        switch (action) {
        case ScrollingLayerPositionAction::Set:
            graphicsLayer->setPosition(position);
            break;

        case ScrollingLayerPositionAction::SetApproximate:
            graphicsLayer->setApproximatePosition(position);
            break;
        
        case ScrollingLayerPositionAction::Sync:
            graphicsLayer->syncPosition(position);
            break;
        }
    }
}

FloatSize ScrollingStateStickyNode::scrollDeltaSinceLastCommit(const LayoutRect& viewportRect) const
{
    auto layerPosition = computeLayerPosition(viewportRect);
    return layerPosition - m_constraints.layerPositionAtLastLayout();
}

void ScrollingStateStickyNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Sticky node";
    ScrollingStateNode::dumpProperties(ts, behavior);

    if (m_constraints.anchorEdges()) {
        TextStream::GroupScope scope(ts);
        ts << "anchor edges: ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeLeft))
            ts << "AnchorEdgeLeft ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeRight))
            ts << "AnchorEdgeRight ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeTop))
            ts << "AnchorEdgeTop ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeBottom))
            ts << "AnchorEdgeBottom";
    }

    if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeLeft))
        ts.dumpProperty("left offset", m_constraints.leftOffset());
    if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeRight))
        ts.dumpProperty("right offset", m_constraints.rightOffset());
    if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeTop))
        ts.dumpProperty("top offset", m_constraints.topOffset());
    if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeBottom))
        ts.dumpProperty("bottom offset", m_constraints.bottomOffset());

    ts.dumpProperty("containing block rect", m_constraints.containingBlockRect());

    ts.dumpProperty("sticky box rect", m_constraints.stickyBoxRect());

    ts.dumpProperty("constraining rect", m_constraints.constrainingRectAtLastLayout());

    ts.dumpProperty("sticky offset at last layout", m_constraints.stickyOffsetAtLastLayout());

    ts.dumpProperty("layer position at last layout", m_constraints.layerPositionAtLastLayout());
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
