/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#import "ScrollingTreeStickyNodeCocoa.h"

#if ENABLE(ASYNC_SCROLLING)

#import "Logging.h"
#import "ScrollingStateStickyNode.h"
#import "ScrollingThread.h"
#import "ScrollingTree.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/TZoneMalloc.h>
#import <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeStickyNodeCocoa);

Ref<ScrollingTreeStickyNodeCocoa> ScrollingTreeStickyNodeCocoa::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeStickyNodeCocoa(scrollingTree, nodeID));
}

ScrollingTreeStickyNodeCocoa::ScrollingTreeStickyNodeCocoa(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeStickyNode(scrollingTree, nodeID)
{
}

bool ScrollingTreeStickyNodeCocoa::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    if (stateNode.hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CALayer*>(stateNode.layer());

    return ScrollingTreeStickyNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreeStickyNodeCocoa::applyLayerPositions()
{
    auto layerPosition = computeLayerPosition();

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeStickyNodeCocoa " << scrollingNodeID() << " constrainingRectAtLastLayout " << m_constraints.constrainingRectAtLastLayout() << " last layer pos " << m_constraints.layerPositionAtLastLayout() << " layerPosition " << layerPosition);

#if ENABLE(SCROLLING_THREAD)
    if (ScrollingThread::isCurrentThread()) {
        // Match the behavior of ScrollingTreeFrameScrollingNodeMac::repositionScrollingLayers().
        if (!scrollingTree()->isScrollingSynchronizedWithMainThread())
            [m_layer _web_setLayerTopLeftPosition:CGPointZero];
    }
#endif

    [m_layer _web_setLayerTopLeftPosition:layerPosition - m_constraints.alignmentOffset()];
}

FloatPoint ScrollingTreeStickyNodeCocoa::layerTopLeft() const
{
    FloatRect layerBounds = [m_layer bounds];
    FloatPoint anchorPoint = [m_layer anchorPoint];
    FloatPoint position = [m_layer position];
    return position - toFloatSize(anchorPoint) * layerBounds.size() + m_constraints.alignmentOffset();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
