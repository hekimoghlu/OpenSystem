/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#include "ScrollingTreeOverflowScrollProxyNodeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "Logging.h"
#include "ScrollingStateOverflowScrollProxyNode.h"
#include "ScrollingStateTree.h"
#include "ScrollingTree.h"

namespace WebCore {

Ref<ScrollingTreeOverflowScrollProxyNodeCoordinated> ScrollingTreeOverflowScrollProxyNodeCoordinated::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeOverflowScrollProxyNodeCoordinated(scrollingTree, nodeID));
}

ScrollingTreeOverflowScrollProxyNodeCoordinated::ScrollingTreeOverflowScrollProxyNodeCoordinated(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeOverflowScrollProxyNode(scrollingTree, nodeID)
{
}

ScrollingTreeOverflowScrollProxyNodeCoordinated::~ScrollingTreeOverflowScrollProxyNodeCoordinated() = default;

bool ScrollingTreeOverflowScrollProxyNodeCoordinated::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CoordinatedPlatformLayer*>(stateNode.layer());

    return ScrollingTreeOverflowScrollProxyNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreeOverflowScrollProxyNodeCoordinated::applyLayerPositions()
{
    FloatPoint scrollOffset = computeLayerPosition();

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeOverflowScrollProxyNodeCoordinated " << scrollingNodeID() << " applyLayerPositions: setting bounds origin to " << scrollOffset);

    m_layer->setBoundsOriginForScrolling(scrollOffset);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
