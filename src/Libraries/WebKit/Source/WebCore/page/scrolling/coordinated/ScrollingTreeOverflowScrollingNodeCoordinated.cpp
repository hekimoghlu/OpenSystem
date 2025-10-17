/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#include "ScrollingTreeOverflowScrollingNodeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "ScrollingTreeScrollingNodeDelegateCoordinated.h"
#include "ThreadedScrollingTree.h"

namespace WebCore {

Ref<ScrollingTreeOverflowScrollingNode> ScrollingTreeOverflowScrollingNodeCoordinated::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeOverflowScrollingNodeCoordinated(scrollingTree, nodeID));
}

ScrollingTreeOverflowScrollingNodeCoordinated::ScrollingTreeOverflowScrollingNodeCoordinated(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeOverflowScrollingNode(scrollingTree, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateCoordinated>(*this, downcast<ThreadedScrollingTree>(scrollingTree).scrollAnimatorEnabled());
}

ScrollingTreeOverflowScrollingNodeCoordinated::~ScrollingTreeOverflowScrollingNodeCoordinated() = default;

ScrollingTreeScrollingNodeDelegateCoordinated& ScrollingTreeOverflowScrollingNodeCoordinated::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateCoordinated*>(m_delegate.get());
}

bool ScrollingTreeOverflowScrollingNodeCoordinated::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreeOverflowScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    m_delegate->updateFromStateNode(downcast<ScrollingStateScrollingNode>(stateNode));
    return true;
}

void ScrollingTreeOverflowScrollingNodeCoordinated::repositionScrollingLayers()
{
    auto* scrollLayer = static_cast<CoordinatedPlatformLayer*>(scrollContainerLayer());
    ASSERT(scrollLayer);

    auto scrollOffset = currentScrollOffset();
    scrollLayer->setBoundsOriginForScrolling(scrollOffset);

    delegate().updateVisibleLengths();
}

WheelEventHandlingResult ScrollingTreeOverflowScrollingNodeCoordinated::handleWheelEvent(const PlatformWheelEvent& wheelEvent, EventTargeting eventTargeting)
{
    if (!canHandleWheelEvent(wheelEvent, eventTargeting))
        return WheelEventHandlingResult::unhandled();

    return WheelEventHandlingResult::result(delegate().handleWheelEvent(wheelEvent));
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
