/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#import "ScrollingTreePluginScrollingNodeMac.h"

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#import "Logging.h"
#import "ScrollingStatePluginScrollingNode.h"
#import "ScrollingTree.h"
#import "ScrollingTreeScrollingNodeDelegateMac.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/BlockObjCExceptions.h>
#import <wtf/text/TextStream.h>

namespace WebCore {

Ref<ScrollingTreePluginScrollingNodeMac> ScrollingTreePluginScrollingNodeMac::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreePluginScrollingNodeMac(scrollingTree, nodeID));
}

ScrollingTreePluginScrollingNodeMac::ScrollingTreePluginScrollingNodeMac(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreePluginScrollingNode(scrollingTree, nodeID)
{
    m_delegate = makeUnique<ScrollingTreeScrollingNodeDelegateMac>(*this);
}

ScrollingTreePluginScrollingNodeMac::~ScrollingTreePluginScrollingNodeMac() = default;

ScrollingTreeScrollingNodeDelegateMac& ScrollingTreePluginScrollingNodeMac::delegate() const
{
    return *static_cast<ScrollingTreeScrollingNodeDelegateMac*>(m_delegate.get());
}

void ScrollingTreePluginScrollingNodeMac::willBeDestroyed()
{
    delegate().nodeWillBeDestroyed();
}

bool ScrollingTreePluginScrollingNodeMac::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (!ScrollingTreePluginScrollingNode::commitStateBeforeChildren(stateNode))
        return false;

    auto* state = dynamicDowncast<ScrollingStatePluginScrollingNode>(stateNode);
    if (!state)
        return false;

    m_delegate->updateFromStateNode(*state);
    return true;
}

WheelEventHandlingResult ScrollingTreePluginScrollingNodeMac::handleWheelEvent(const PlatformWheelEvent& wheelEvent, EventTargeting eventTargeting)
{
#if ENABLE(SCROLLING_THREAD)
    if (hasNonRepaintSynchronousScrollingReasons() && eventTargeting != EventTargeting::NodeOnly)
        return { { WheelEventProcessingSteps::SynchronousScrolling, WheelEventProcessingSteps::NonBlockingDOMEventDispatch }, false };
#endif

    if (!canHandleWheelEvent(wheelEvent, eventTargeting))
        return WheelEventHandlingResult::unhandled();

    return WheelEventHandlingResult::result(delegate().handleWheelEvent(wheelEvent));
}

void ScrollingTreePluginScrollingNodeMac::willDoProgrammaticScroll(const FloatPoint& targetScrollPosition)
{
    delegate().willDoProgrammaticScroll(targetScrollPosition);
}

void ScrollingTreePluginScrollingNodeMac::currentScrollPositionChanged(ScrollType scrollType, ScrollingLayerPositionAction action)
{
    ScrollingTreePluginScrollingNode::currentScrollPositionChanged(scrollType, action);
    delegate().currentScrollPositionChanged();
}

void ScrollingTreePluginScrollingNodeMac::repositionScrollingLayers()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [static_cast<CALayer*>(scrollContainerLayer()) _web_setLayerBoundsOrigin:currentScrollOffset()];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollingTreePluginScrollingNodeMac::repositionRelatedLayers()
{
    delegate().updateScrollbarPainters();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)
