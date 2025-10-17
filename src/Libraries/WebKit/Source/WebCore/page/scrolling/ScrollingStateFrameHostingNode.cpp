/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include "ScrollingStateFrameHostingNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingStateFrameHostingNode);

Ref<ScrollingStateFrameHostingNode> ScrollingStateFrameHostingNode::create(ScrollingStateTree& stateTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingStateFrameHostingNode(stateTree, nodeID));
}

Ref<ScrollingStateFrameHostingNode> ScrollingStateFrameHostingNode::create(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID, std::optional<LayerHostingContextIdentifier> identifier)
{
    return adoptRef(*new ScrollingStateFrameHostingNode(nodeID, WTFMove(children), changedProperties, layerID, identifier));
}

ScrollingStateFrameHostingNode::ScrollingStateFrameHostingNode(ScrollingNodeID nodeID, Vector<Ref<ScrollingStateNode>>&& children, OptionSet<ScrollingStateNodeProperty> changedProperties, std::optional<PlatformLayerIdentifier> layerID, std::optional<LayerHostingContextIdentifier> identifier)
    : ScrollingStateNode(ScrollingNodeType::FrameHosting, nodeID, WTFMove(children), changedProperties, layerID)
    , m_hostingContext(identifier)
{
    ASSERT(isFrameHostingNode());
}

ScrollingStateFrameHostingNode::ScrollingStateFrameHostingNode(ScrollingStateTree& stateTree, ScrollingNodeID nodeID)
    : ScrollingStateNode(ScrollingNodeType::FrameHosting, stateTree, nodeID)
{
    ASSERT(isFrameHostingNode());
}

ScrollingStateFrameHostingNode::ScrollingStateFrameHostingNode(const ScrollingStateFrameHostingNode& stateNode, ScrollingStateTree& adoptiveTree)
    : ScrollingStateNode(stateNode, adoptiveTree)
    , m_hostingContext(stateNode.layerHostingContextIdentifier())
{
}

ScrollingStateFrameHostingNode::~ScrollingStateFrameHostingNode() = default;

void ScrollingStateFrameHostingNode::setLayerHostingContextIdentifier(const std::optional<LayerHostingContextIdentifier> identifier)
{
    if (identifier == m_hostingContext)
        return;
    m_hostingContext = identifier;
    setPropertyChanged(Property::LayerHostingContextIdentifier);
}

Ref<ScrollingStateNode> ScrollingStateFrameHostingNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStateFrameHostingNode(*this, adoptiveTree));
}

void ScrollingStateFrameHostingNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Frame hosting node";
    ScrollingStateNode::dumpProperties(ts, behavior);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
