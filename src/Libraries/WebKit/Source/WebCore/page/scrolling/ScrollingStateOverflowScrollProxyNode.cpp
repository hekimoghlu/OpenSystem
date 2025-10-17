/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "ScrollingStateOverflowScrollProxyNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateOverflowScrollingNode.h"
#include "ScrollingStateTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingStateOverflowScrollProxyNode);

ScrollingStateOverflowScrollProxyNode::ScrollingStateOverflowScrollProxyNode(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID, std::optional<ScrollingNodeID> overflowScrollingNode)
    : ScrollingStateNode(ScrollingNodeType::OverflowProxy, nodeID, WTFMove(children), changedProperties, layerID)
    , m_overflowScrollingNodeID(overflowScrollingNode)
{
}

ScrollingStateOverflowScrollProxyNode::ScrollingStateOverflowScrollProxyNode(ScrollingStateTree& stateTree, ScrollingNodeID nodeID)
    : ScrollingStateNode(ScrollingNodeType::OverflowProxy, stateTree, nodeID)
{
}

ScrollingStateOverflowScrollProxyNode::ScrollingStateOverflowScrollProxyNode(const ScrollingStateOverflowScrollProxyNode& stateNode, ScrollingStateTree& adoptiveTree)
    : ScrollingStateNode(stateNode, adoptiveTree)
    , m_overflowScrollingNodeID(stateNode.overflowScrollingNode())
{
}

ScrollingStateOverflowScrollProxyNode::~ScrollingStateOverflowScrollProxyNode() = default;

Ref<ScrollingStateNode> ScrollingStateOverflowScrollProxyNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStateOverflowScrollProxyNode(*this, adoptiveTree));
}

OptionSet<ScrollingStateNode::Property> ScrollingStateOverflowScrollProxyNode::applicableProperties() const
{
    constexpr OptionSet<Property> nodeProperties = { Property::OverflowScrollingNode };

    auto properties = ScrollingStateNode::applicableProperties();
    properties.add(nodeProperties);
    return properties;
}


void ScrollingStateOverflowScrollProxyNode::setOverflowScrollingNode(std::optional<ScrollingNodeID> nodeID)
{
    if (nodeID == m_overflowScrollingNodeID)
        return;
    
    m_overflowScrollingNodeID = nodeID;
    setPropertyChanged(Property::OverflowScrollingNode);
}

void ScrollingStateOverflowScrollProxyNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Overflow scroll proxy node";
    
    ScrollingStateNode::dumpProperties(ts, behavior);

    if (auto relatedOverflowNode = scrollingStateTree().stateNodeForID(m_overflowScrollingNodeID)) {
        if (RefPtr overflowScrollingNode = dynamicDowncast<ScrollingStateOverflowScrollingNode>(relatedOverflowNode))
            ts.dumpProperty("related overflow scrolling node scroll position", overflowScrollingNode->scrollPosition());
    }

    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeNodeIDs)
        ts.dumpProperty("overflow scrolling node", overflowScrollingNode());
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
