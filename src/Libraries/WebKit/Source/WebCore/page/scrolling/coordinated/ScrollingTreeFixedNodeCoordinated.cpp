/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#include "ScrollingTreeFixedNodeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "Logging.h"
#include "ScrollingStateFixedNode.h"
#include "ScrollingThread.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<ScrollingTreeFixedNodeCoordinated> ScrollingTreeFixedNodeCoordinated::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFixedNodeCoordinated(scrollingTree, nodeID));
}

ScrollingTreeFixedNodeCoordinated::ScrollingTreeFixedNodeCoordinated(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeFixedNode(scrollingTree, nodeID)
{
}

ScrollingTreeFixedNodeCoordinated::~ScrollingTreeFixedNodeCoordinated() = default;

bool ScrollingTreeFixedNodeCoordinated::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!is<ScrollingStateFixedNode>(stateNode))
        return false;

    auto& fixedStateNode = downcast<ScrollingStateFixedNode>(stateNode);
    if (fixedStateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CoordinatedPlatformLayer*>(fixedStateNode.layer());

    return ScrollingTreeFixedNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreeFixedNodeCoordinated::applyLayerPositions()
{
    auto layerPosition = computeLayerPosition();

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeFixedNode " << scrollingNodeID() << " relatedNodeScrollPositionDidChange: viewportRectAtLastLayout " << m_constraints.viewportRectAtLastLayout() << " last layer pos " << m_constraints.layerPositionAtLastLayout() << " layerPosition " << layerPosition);

    // Match the behavior of ScrollingTreeFrameScrollingNodeCoordinated::repositionScrollingLayers().
    CoordinatedPlatformLayer::ForcePositionSync forceSync = ScrollingThread::isCurrentThread() && !scrollingTree()->isScrollingSynchronizedWithMainThread() ?
        CoordinatedPlatformLayer::ForcePositionSync::Yes : CoordinatedPlatformLayer::ForcePositionSync::No;

    m_layer->setTopLeftPositionForScrolling(layerPosition - m_constraints.alignmentOffset(), forceSync);
}

void ScrollingTreeFixedNodeCoordinated::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ScrollingTreeFixedNode::dumpProperties(ts, behavior);

    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeLayerPositions) {
        FloatPoint layerTopLeft = m_layer->topLeftPositionForScrolling() + m_constraints.alignmentOffset();
        ts.dumpProperty("layer top left", layerTopLeft);
    }
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
