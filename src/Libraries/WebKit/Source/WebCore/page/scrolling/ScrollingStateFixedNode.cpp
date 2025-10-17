/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#include "ScrollingStateFixedNode.h"

#include "GraphicsLayer.h"
#include "Logging.h"
#include "ScrollingStateTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

#if ENABLE(ASYNC_SCROLLING)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingStateFixedNode);

ScrollingStateFixedNode::ScrollingStateFixedNode(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID, FixedPositionViewportConstraints&& constraints)
    : ScrollingStateNode(ScrollingNodeType::Fixed, nodeID, WTFMove(children), changedProperties, layerID)
    , m_constraints(WTFMove(constraints))
{
}

ScrollingStateFixedNode::ScrollingStateFixedNode(const ScrollingStateFixedNode& node, ScrollingStateTree& adoptiveTree)
    : ScrollingStateNode(node, adoptiveTree)
    , m_constraints(FixedPositionViewportConstraints(node.viewportConstraints()))
{
}

ScrollingStateFixedNode::ScrollingStateFixedNode(ScrollingStateTree& tree, ScrollingNodeID nodeID)
    : ScrollingStateNode(ScrollingNodeType::Fixed, tree, nodeID)
{
}

ScrollingStateFixedNode::~ScrollingStateFixedNode() = default;

Ref<ScrollingStateNode> ScrollingStateFixedNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStateFixedNode(*this, adoptiveTree));
}

OptionSet<ScrollingStateNode::Property> ScrollingStateFixedNode::applicableProperties() const
{
    constexpr OptionSet<Property> nodeProperties = { Property::ViewportConstraints };

    auto properties = ScrollingStateNode::applicableProperties();
    properties.add(nodeProperties);
    return properties;
}

void ScrollingStateFixedNode::updateConstraints(const FixedPositionViewportConstraints& constraints)
{
    if (m_constraints == constraints)
        return;

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingStateFixedNode " << scrollingNodeID() << " updateConstraints with viewport rect " << constraints.viewportRectAtLastLayout() << " layer pos at last layout " << constraints.layerPositionAtLastLayout() << " offset from top " << (constraints.layerPositionAtLastLayout().y() - constraints.viewportRectAtLastLayout().y()));

    m_constraints = constraints;
    setPropertyChanged(Property::ViewportConstraints);
}

void ScrollingStateFixedNode::reconcileLayerPositionForViewportRect(const LayoutRect& viewportRect, ScrollingLayerPositionAction action)
{
    FloatPoint position = m_constraints.layerPositionForViewportRect(viewportRect);
    if (layer().representsGraphicsLayer()) {
        RefPtr graphicsLayer = static_cast<GraphicsLayer*>(layer());
        ASSERT(graphicsLayer);
        // Crash data suggest that graphicsLayer can be null: rdar://105887621.
        if (!graphicsLayer)
            return;

        LOG_WITH_STREAM(Scrolling, stream << "ScrollingStateFixedNode " << scrollingNodeID() <<" reconcileLayerPositionForViewportRect " << action << " position of layer " << graphicsLayer->primaryLayerID() << " to " << position);
        
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

void ScrollingStateFixedNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Fixed node";
    ScrollingStateNode::dumpProperties(ts, behavior);

    if (m_constraints.anchorEdges()) {
        TextStream::GroupScope scope(ts);
        ts << "anchor edges: ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeLeft))
            ts << "AnchorEdgeLeft ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeRight))
            ts << "AnchorEdgeRight ";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeTop))
            ts << "AnchorEdgeTop";
        if (m_constraints.hasAnchorEdge(ViewportConstraints::AnchorEdgeBottom))
            ts << "AnchorEdgeBottom";
    }

    if (!m_constraints.alignmentOffset().isEmpty())
        ts.dumpProperty("alignment offset", m_constraints.alignmentOffset());

    if (!m_constraints.viewportRectAtLastLayout().isEmpty())
        ts.dumpProperty("viewport rect at last layout", m_constraints.viewportRectAtLastLayout());

    if (m_constraints.layerPositionAtLastLayout() != FloatPoint())
        ts.dumpProperty("layer position at last layout", m_constraints.layerPositionAtLastLayout());
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
