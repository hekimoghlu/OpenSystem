/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#include "ScrollingTreePluginScrollingNodeRemoteMac.h"

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#include "RemoteScrollingTree.h"
#include <WebCore/ScrollingStatePluginScrollingNode.h>
#include <WebCore/ScrollingTreeScrollingNodeDelegate.h>

namespace WebKit {
using namespace WebCore;

Ref<ScrollingTreePluginScrollingNodeRemoteMac> ScrollingTreePluginScrollingNodeRemoteMac::create(ScrollingTree& tree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreePluginScrollingNodeRemoteMac(tree, nodeID));
}

ScrollingTreePluginScrollingNodeRemoteMac::ScrollingTreePluginScrollingNodeRemoteMac(ScrollingTree& tree, ScrollingNodeID nodeID)
    : ScrollingTreePluginScrollingNodeMac(tree, nodeID)
{
    m_delegate->initScrollbars();
}

ScrollingTreePluginScrollingNodeRemoteMac::~ScrollingTreePluginScrollingNodeRemoteMac() = default;

void ScrollingTreePluginScrollingNodeRemoteMac::repositionRelatedLayers()
{
    ScrollingTreePluginScrollingNodeMac::repositionRelatedLayers();
    m_delegate->updateScrollbarLayers();
}

void ScrollingTreePluginScrollingNodeRemoteMac::handleWheelEventPhase(const PlatformWheelEventPhase phase)
{
    m_delegate->handleWheelEventPhase(phase);
}

String ScrollingTreePluginScrollingNodeRemoteMac::scrollbarStateForOrientation(ScrollbarOrientation orientation) const
{
    return m_delegate->scrollbarStateForOrientation(orientation);
}

}

#endif
