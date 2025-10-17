/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "ScrollingStatePositionedNode.h"

#include "GraphicsLayer.h"
#include "Logging.h"
#include "ScrollingStateTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

#if ENABLE(ASYNC_SCROLLING)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingStatePositionedNode);

ScrollingStatePositionedNode::ScrollingStatePositionedNode(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID, Vector<ScrollingNodeID>&& relatedOverflowScrollingNodes, AbsolutePositionConstraints&& constraints)
    : ScrollingStateNode(ScrollingNodeType::Positioned, nodeID, WTFMove(children), changedProperties, layerID)
    , m_relatedOverflowScrollingNodes(WTFMove(relatedOverflowScrollingNodes))
    , m_constraints(WTFMove(constraints))
{
}

ScrollingStatePositionedNode::ScrollingStatePositionedNode(ScrollingStateTree& tree, ScrollingNodeID nodeID)
    : ScrollingStateNode(ScrollingNodeType::Positioned, tree, nodeID)
{
}

ScrollingStatePositionedNode::ScrollingStatePositionedNode(const ScrollingStatePositionedNode& node, ScrollingStateTree& adoptiveTree)
    : ScrollingStateNode(node, adoptiveTree)
    , m_relatedOverflowScrollingNodes(node.relatedOverflowScrollingNodes())
    , m_constraints(node.layoutConstraints())
{
}

ScrollingStatePositionedNode::~ScrollingStatePositionedNode() = default;

Ref<ScrollingStateNode> ScrollingStatePositionedNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStatePositionedNode(*this, adoptiveTree));
}

OptionSet<ScrollingStateNode::Property> ScrollingStatePositionedNode::applicableProperties() const
{
    constexpr OptionSet<Property> nodeProperties = { Property::RelatedOverflowScrollingNodes, Property::LayoutConstraintData };

    auto properties = ScrollingStateNode::applicableProperties();
    properties.add(nodeProperties);
    return properties;
}

void ScrollingStatePositionedNode::setRelatedOverflowScrollingNodes(Vector<ScrollingNodeID>&& nodes)
{
    if (nodes == m_relatedOverflowScrollingNodes)
        return;

    m_relatedOverflowScrollingNodes = WTFMove(nodes);
    setPropertyChanged(Property::RelatedOverflowScrollingNodes);
}

void ScrollingStatePositionedNode::updateConstraints(const AbsolutePositionConstraints& constraints)
{
    if (m_constraints == constraints)
        return;

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingStatePositionedNode " << scrollingNodeID() << " updateConstraints " << constraints);

    m_constraints = constraints;
    setPropertyChanged(Property::LayoutConstraintData);
}

void ScrollingStatePositionedNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Positioned node";
    ScrollingStateNode::dumpProperties(ts, behavior);

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
