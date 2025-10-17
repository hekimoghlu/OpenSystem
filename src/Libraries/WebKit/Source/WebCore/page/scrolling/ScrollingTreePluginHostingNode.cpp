/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "ScrollingTreePluginHostingNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "Logging.h"
#include "ScrollingStatePluginHostingNode.h"
#include "ScrollingStateTree.h"
#include "ScrollingTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreePluginHostingNode);

Ref<ScrollingTreePluginHostingNode> ScrollingTreePluginHostingNode::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreePluginHostingNode(scrollingTree, nodeID));
}

ScrollingTreePluginHostingNode::ScrollingTreePluginHostingNode(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeNode(scrollingTree, ScrollingNodeType::PluginHosting, nodeID)
{
    ASSERT(isPluginHostingNode());
}

ScrollingTreePluginHostingNode::~ScrollingTreePluginHostingNode() = default;

bool ScrollingTreePluginHostingNode::commitStateBeforeChildren(const ScrollingStateNode&)
{
    return true;
}

void ScrollingTreePluginHostingNode::applyLayerPositions()
{
}

void ScrollingTreePluginHostingNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "plugin hosting node";
    ScrollingTreeNode::dumpProperties(ts, behavior);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
