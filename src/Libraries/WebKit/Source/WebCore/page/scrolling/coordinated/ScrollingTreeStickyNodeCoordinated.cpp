/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "ScrollingTreeStickyNodeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "Logging.h"
#include "ScrollingStateStickyNode.h"
#include "ScrollingThread.h"
#include "ScrollingTree.h"
#include "ScrollingTreeFixedNode.h"
#include "ScrollingTreeFrameScrollingNode.h"
#include "ScrollingTreeOverflowScrollingNode.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<ScrollingTreeStickyNodeCoordinated> ScrollingTreeStickyNodeCoordinated::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeStickyNodeCoordinated(scrollingTree, nodeID));
}

ScrollingTreeStickyNodeCoordinated::ScrollingTreeStickyNodeCoordinated(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeStickyNode(scrollingTree, nodeID)
{
}

bool ScrollingTreeStickyNodeCoordinated::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CoordinatedPlatformLayer*>(stateNode.layer());

    return ScrollingTreeStickyNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreeStickyNodeCoordinated::applyLayerPositions()
{
    auto layerPosition = computeLayerPosition();

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeStickyNodeCoordinated " << scrollingNodeID() << " constrainingRectAtLastLayout " << m_constraints.constrainingRectAtLastLayout() << " last layer pos " << m_constraints.layerPositionAtLastLayout() << " layerPosition " << layerPosition);

    // Match the behavior of ScrollingTreeFrameScrollingNodeCoordinated::repositionScrollingLayers().
    CoordinatedPlatformLayer::ForcePositionSync forceSync = ScrollingThread::isCurrentThread() && !scrollingTree()->isScrollingSynchronizedWithMainThread() ?
        CoordinatedPlatformLayer::ForcePositionSync::Yes : CoordinatedPlatformLayer::ForcePositionSync::No;

    m_layer->setTopLeftPositionForScrolling(layerPosition - m_constraints.alignmentOffset(), forceSync);
}

FloatPoint ScrollingTreeStickyNodeCoordinated::layerTopLeft() const
{
    return m_layer->topLeftPositionForScrolling() + m_constraints.alignmentOffset();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
