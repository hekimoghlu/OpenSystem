/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
#import "config.h"
#import "ScrollingTreeOverflowScrollingNodeMac.h"

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#import "Logging.h"
#import "ScrollingStateOverflowScrollingNode.h"
#import "ScrollingTree.h"
#import "ScrollingTreeScrollingNodeDelegateMac.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/BlockObjCExceptions.h>
#import <wtf/text/TextStream.h>

namespace WebCore {

Ref<ScrollingTreeOverflowScrollingNodeMac> ScrollingTreeOverflowScrollingNodeMac::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeOverflowScrollingNodeMac(scrollingTree, nodeID));
}

ScrollingTreeOverflowScrollingNodeMac::ScrollingTreeOverflowScrollingNodeMac(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeOverflowScrollingNode(scrollingTree, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateMac>(*this);
}

ScrollingTreeOverflowScrollingNodeMac::~ScrollingTreeOverflowScrollingNodeMac() = default;

ScrollingTreeScrollingNodeDelegateMac& ScrollingTreeOverflowScrollingNodeMac::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateMac*>(m_delegate.get());
}

void ScrollingTreeOverflowScrollingNodeMac::willBeDestroyed()
{
    delegate().nodeWillBeDestroyed();
}

bool ScrollingTreeOverflowScrollingNodeMac::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreeOverflowScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    auto* state = dynamicDowncast<ScrollingStateOverflowScrollingNode>(stateNode);
    if (!state)
        return false;

    m_delegate->updateFromStateNode(*state);
    return true;
}

WheelEventHandlingResult ScrollingTreeOverflowScrollingNodeMac::handleWheelEvent(const PlatformWheelEvent& wheelEvent, EventTargeting eventTargeting)
{
#if ENABLE(SCROLLING_THREAD)
    if (hasNonRepaintSynchronousScrollingReasons() && eventTargeting != EventTargeting::NodeOnly)
        return { { WheelEventProcessingSteps::SynchronousScrolling, WheelEventProcessingSteps::NonBlockingDOMEventDispatch }, false };
#endif

    if (!canHandleWheelEvent(wheelEvent, eventTargeting))
        return WheelEventHandlingResult::unhandled();

    return WheelEventHandlingResult::result(delegate().handleWheelEvent(wheelEvent));
}

void ScrollingTreeOverflowScrollingNodeMac::willDoProgrammaticScroll(const FloatPoint& targetScrollPosition)
{
    delegate().willDoProgrammaticScroll(targetScrollPosition);
}

void ScrollingTreeOverflowScrollingNodeMac::currentScrollPositionChanged(ScrollType scrollType, ScrollingLayerPositionAction action)
{
    ScrollingTreeOverflowScrollingNode::currentScrollPositionChanged(scrollType, action);
    delegate().currentScrollPositionChanged();
}

void ScrollingTreeOverflowScrollingNodeMac::repositionScrollingLayers()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [static_cast<CALayer*>(scrollContainerLayer()) _web_setLayerBoundsOrigin:currentScrollOffset()];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollingTreeOverflowScrollingNodeMac::repositionRelatedLayers()
{
    delegate().updateScrollbarPainters();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)
