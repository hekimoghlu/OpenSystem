/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#import "ScrollingTreeFixedNodeCocoa.h"

#if ENABLE(ASYNC_SCROLLING)

#import "Logging.h"
#import "ScrollingStateFixedNode.h"
#import "ScrollingThread.h"
#import "WebCoreCALayerExtras.h"
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollingTreeFixedNodeCocoa);

Ref<ScrollingTreeFixedNodeCocoa> ScrollingTreeFixedNodeCocoa::create(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
{
    return adoptRef(*new ScrollingTreeFixedNodeCocoa(scrollingTree, nodeID));
}

ScrollingTreeFixedNodeCocoa::ScrollingTreeFixedNodeCocoa(ScrollingTree& scrollingTree, ScrollingNodeID nodeID)
    : ScrollingTreeFixedNode(scrollingTree, nodeID)
{
}

ScrollingTreeFixedNodeCocoa::~ScrollingTreeFixedNodeCocoa() = default;

bool ScrollingTreeFixedNodeCocoa::commitStateBeforeChildren(const ScrollingStateNode& stateNode)
{
    auto* fixedStateNode = dynamicDowncast<ScrollingStateFixedNode>(stateNode);
    if (!fixedStateNode)
        return false;

    if (fixedStateNode->hasChangedProperty(ScrollingStateNode::Property::Layer))
        m_layer = static_cast<CALayer*>(fixedStateNode->layer());

    return ScrollingTreeFixedNode::commitStateBeforeChildren(stateNode);
}

void ScrollingTreeFixedNodeCocoa::applyLayerPositions()
{
    auto layerPosition = computeLayerPosition();

    LOG_WITH_STREAM(Scrolling, stream << "ScrollingTreeFixedNode " << scrollingNodeID() << " relatedNodeScrollPositionDidChange: viewportRectAtLastLayout " << m_constraints.viewportRectAtLastLayout() << " last layer pos " << m_constraints.layerPositionAtLastLayout() << " layerPosition " << layerPosition);

#if ENABLE(SCROLLING_THREAD)
    if (ScrollingThread::isCurrentThread()) {
        // Match the behavior of ScrollingTreeFrameScrollingNodeMac::repositionScrollingLayers().
        if (!scrollingTree()->isScrollingSynchronizedWithMainThread())
            [m_layer _web_setLayerTopLeftPosition:CGPointZero];
    }
#endif

    [m_layer _web_setLayerTopLeftPosition:layerPosition - m_constraints.alignmentOffset()];
}

void ScrollingTreeFixedNodeCocoa::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ScrollingTreeFixedNode::dumpProperties(ts, behavior);

    if (behavior & ScrollingStateTreeAsTextBehavior::IncludeLayerPositions) {
        FloatRect layerBounds = [m_layer bounds];
        FloatPoint anchorPoint = [m_layer anchorPoint];
        FloatPoint position = [m_layer position];
        FloatPoint layerTopLeft = position - toFloatSize(anchorPoint) * layerBounds.size() + m_constraints.alignmentOffset();
        ts.dumpProperty("layer top left", layerTopLeft);
    }
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
