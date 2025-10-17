/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#include "ScrollingTreePositionedNodeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "Logging.h"
#include "ScrollingStatePositionedNode.h"
#include "ScrollingThread.h"
#include "ScrollingTree.h"

namespace WebCore {

Ref<ScrollingTreePositionedNodeCoordinated> ScrollingTreePositionedNodeCoordinated::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreePositionedNodeCoordinated(scrollingTree, nodeID));
}

ScrollingTreePositionedNodeCoordinated::ScrollingTreePositionedNodeCoordinated(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreePositionedNode(scrollingTree, nodeID)
{
}

ScrollingTreePositionedNodeCoordinated::~ScrollingTreePositionedNodeCoordinated() = default;

bool ScrollingTreePositionedNodeCoordinated::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CoordinatedPlatformLayer*>(stateNode.layer());

    return ScrollingTreePositionedNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreePositionedNodeCoordinated::applyLayerPositions()
{
    FloatSize delta = scrollDeltaSinceLastCommit();
    FloatPoint layerPosition = m_constraints.layerPositionAtLastLayout() - delta;

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreePositionedNode " << scrollingNodeID() << " applyLayerPositions: overflow delta " << delta << " moving layer to " << layerPosition);

    // Match the behavior of ScrollingTreeFrameScrollingNodeCoordinated::repositionScrollingLayers().
    CoordinatedPlatformLayer::ForcePositionSync forceSync = ScrollingThread::isCurrentThread() && !scrollingTree()->isScrollingSynchronizedWithMainThread() ?
        CoordinatedPlatformLayer::ForcePositionSync::Yes : CoordinatedPlatformLayer::ForcePositionSync::No;

    m_layer->setTopLeftPositionForScrolling(layerPosition - m_constraints.alignmentOffset(), forceSync);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
