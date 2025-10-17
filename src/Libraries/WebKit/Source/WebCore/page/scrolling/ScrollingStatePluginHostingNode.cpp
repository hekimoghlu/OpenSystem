/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
#include "ScrollingStatePluginHostingNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingStatePluginHostingNode);

Ref<ScrollingStatePluginHostingNode> ScrollingStatePluginHostingNode::create(ScrollingStateTree& stateTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingStatePluginHostingNode(stateTree, nodeID));
}

Ref<ScrollingStatePluginHostingNode> ScrollingStatePluginHostingNode::create(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID)
{
    return adoptRef(*new ScrollingStatePluginHostingNode(nodeID, WTFMove(children), changedProperties, layerID));
}

ScrollingStatePluginHostingNode::ScrollingStatePluginHostingNode(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID)
    : ScrollingStateNode(ScrollingNodeType::PluginHosting, nodeID, WTFMove(children), changedProperties, layerID)
{
    ASSERT(isPluginHostingNode());
}

ScrollingStatePluginHostingNode::ScrollingStatePluginHostingNode(ScrollingStateTree& stateTree, ScrollingNodeID nodeID)
    : ScrollingStateNode(ScrollingNodeType::PluginHosting, stateTree, nodeID)
{
    ASSERT(isPluginHostingNode());
}

ScrollingStatePluginHostingNode::ScrollingStatePluginHostingNode(const ScrollingStatePluginHostingNode& stateNode, ScrollingStateTree& adoptiveTree)
    : ScrollingStateNode(stateNode, adoptiveTree)
{
}

ScrollingStatePluginHostingNode::~ScrollingStatePluginHostingNode() = default;

Ref<ScrollingStateNode> ScrollingStatePluginHostingNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStatePluginHostingNode(*this, adoptiveTree));
}

void ScrollingStatePluginHostingNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Plugin hosting node";
    ScrollingStateNode::dumpProperties(ts, behavior);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
