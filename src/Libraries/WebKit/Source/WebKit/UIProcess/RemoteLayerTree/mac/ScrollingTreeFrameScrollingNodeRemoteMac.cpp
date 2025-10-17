/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#include "ScrollingTreeFrameScrollingNodeRemoteMac.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(ASYNC_SCROLLING) && PLATFORM(MAC)

#include "RemoteScrollingTree.h"

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeFrameScrollingNodeRemoteMac);

using namespace WebCore;

ScrollingTreeFrameScrollingNodeRemoteMac::ScrollingTreeFrameScrollingNodeRemoteMac(ScrollingTree& tree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
    : ScrollingTreeFrameScrollingNodeMac(tree, nodeType, nodeID)
{
    m_delegate->initScrollbars();
}

ScrollingTreeFrameScrollingNodeRemoteMac::~ScrollingTreeFrameScrollingNodeRemoteMac()
{
}

Ref<ScrollingTreeFrameScrollingNodeRemoteMac> ScrollingTreeFrameScrollingNodeRemoteMac::create(ScrollingTree& tree, ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFrameScrollingNodeRemoteMac(tree, nodeType, nodeID));
}

void ScrollingTreeFrameScrollingNodeRemoteMac::repositionRelatedLayers()
{
    ScrollingTreeFrameScrollingNodeMac::repositionRelatedLayers();
    m_delegate->updateScrollbarLayers();
}

void ScrollingTreeFrameScrollingNodeRemoteMac::handleWheelEventPhase(const PlatformWheelEventPhase phase)
{
    m_delegate->handleWheelEventPhase(phase);
}

void ScrollingTreeFrameScrollingNodeRemoteMac::viewWillStartLiveResize()
{
    m_delegate->viewWillStartLiveResize();
}

void ScrollingTreeFrameScrollingNodeRemoteMac::viewWillEndLiveResize()
{
    m_delegate->viewWillEndLiveResize();
}

void ScrollingTreeFrameScrollingNodeRemoteMac::viewSizeDidChange()
{
    m_delegate->viewSizeDidChange();
}

String ScrollingTreeFrameScrollingNodeRemoteMac::scrollbarStateForOrientation(ScrollbarOrientation orientation) const
{
    return m_delegate->scrollbarStateForOrientation(orientation);
}

}

#endif
