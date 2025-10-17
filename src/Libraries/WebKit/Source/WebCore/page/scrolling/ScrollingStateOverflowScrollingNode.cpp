/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
#include "ScrollingStateOverflowScrollingNode.h"

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateTree.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

ScrollingStateOverflowScrollingNode::ScrollingStateOverflowScrollingNode(
    ScrollingNodeID scrollingNodeID,
    Vector<Ref<WebCore::ScrollingStateNode>>&& children,
    OptionSet<ScrollingStateNodeProperty> changedProperties,
    std::optional<WebCore::PlatformLayerIdentifier> layerID,
    FloatSize scrollableAreaSize,
    FloatSize totalContentsSize,
    FloatSize reachableContentsSize,
    FloatPoint scrollPosition,
    IntPoint scrollOrigin,
    ScrollableAreaParameters&& scrollableAreaParameters,
#if ENABLE(SCROLLING_THREAD)
    OptionSet<SynchronousScrollingReason> synchronousScrollingReasons,
#endif
    RequestedScrollData&& requestedScrollData,
    FloatScrollSnapOffsetsInfo&& snapOffsetsInfo,
    std::optional<unsigned> currentHorizontalSnapPointIndex,
    std::optional<unsigned> currentVerticalSnapPointIndex,
    bool isMonitoringWheelEvents,
    std::optional<PlatformLayerIdentifier> scrollContainerLayer,
    std::optional<PlatformLayerIdentifier> scrolledContentsLayer,
    std::optional<PlatformLayerIdentifier> horizontalScrollbarLayer,
    std::optional<PlatformLayerIdentifier> verticalScrollbarLayer,
    bool mouseIsOverContentArea,
    MouseLocationState&& mouseLocationState,
    ScrollbarHoverState&& scrollbarHoverState,
    ScrollbarEnabledState&& scrollbarEnabledState,
    UserInterfaceLayoutDirection scrollbarLayoutDirection,
    ScrollbarWidth scrollbarWidth,
    bool useDarkAppearanceForScrollbars,
    RequestedKeyboardScrollData&& scrollData
) : ScrollingStateScrollingNode(
    ScrollingNodeType::Overflow,
    scrollingNodeID,
    WTFMove(children),
    changedProperties,
    layerID,
    scrollableAreaSize,
    totalContentsSize,
    reachableContentsSize,
    scrollPosition,
    scrollOrigin,
    WTFMove(scrollableAreaParameters),
#if ENABLE(SCROLLING_THREAD)
    synchronousScrollingReasons,
#endif
    WTFMove(requestedScrollData),
    WTFMove(snapOffsetsInfo),
    currentHorizontalSnapPointIndex,
    currentVerticalSnapPointIndex,
    isMonitoringWheelEvents,
    scrollContainerLayer,
    scrolledContentsLayer,
    horizontalScrollbarLayer,
    verticalScrollbarLayer,
    mouseIsOverContentArea,
    WTFMove(mouseLocationState),
    WTFMove(scrollbarHoverState),
    WTFMove(scrollbarEnabledState),
    scrollbarLayoutDirection,
    scrollbarWidth,
    useDarkAppearanceForScrollbars,
    WTFMove(scrollData)
) { }

ScrollingStateOverflowScrollingNode::ScrollingStateOverflowScrollingNode(ScrollingStateTree& stateTree, ScrollingNodeID nodeID)
    : ScrollingStateScrollingNode(stateTree, ScrollingNodeType::Overflow, nodeID)
{
}

ScrollingStateOverflowScrollingNode::ScrollingStateOverflowScrollingNode(const ScrollingStateOverflowScrollingNode& stateNode, ScrollingStateTree& adoptiveTree)
    : ScrollingStateScrollingNode(stateNode, adoptiveTree)
{
}

ScrollingStateOverflowScrollingNode::~ScrollingStateOverflowScrollingNode() = default;

Ref<ScrollingStateNode> ScrollingStateOverflowScrollingNode::clone(ScrollingStateTree& adoptiveTree)
{
    return adoptRef(*new ScrollingStateOverflowScrollingNode(*this, adoptiveTree));
}

void ScrollingStateOverflowScrollingNode::dumpProperties(TextStream& ts, OptionSet<ScrollingStateTreeAsTextBehavior> behavior) const
{
    ts << "Overflow scrolling node";
    
    ScrollingStateScrollingNode::dumpProperties(ts, behavior);
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
