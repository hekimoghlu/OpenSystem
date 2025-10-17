/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#pragma once

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateScrollingNode.h"

namespace WebCore {

class ScrollingStateOverflowScrollingNode : public ScrollingStateScrollingNode {
public:
    template<typename... Args> static Ref<ScrollingStateOverflowScrollingNode> create(Args&&... args) { return adoptRef(*new ScrollingStateOverflowScrollingNode(std::forward<Args>(args)...)); }

    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    virtual ~ScrollingStateOverflowScrollingNode();

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

private:
    WEBCORE_EXPORT ScrollingStateOverflowScrollingNode(
        ScrollingNodeID,
        Vector<Ref<WebCore::ScrollingStateNode>>&& children,
        OptionSet<ScrollingStateNodeProperty> changedProperties,
        std::optional<WebCore::PlatformLayerIdentifier>,
        FloatSize scrollableAreaSize,
        FloatSize totalContentsSize,
        FloatSize reachableContentsSize,
        FloatPoint scrollPosition,
        IntPoint scrollOrigin,
        ScrollableAreaParameters&&,
#if ENABLE(SCROLLING_THREAD)
        OptionSet<SynchronousScrollingReason>,
#endif
        RequestedScrollData&&,
        FloatScrollSnapOffsetsInfo&&,
        std::optional<unsigned> currentHorizontalSnapPointIndex,
        std::optional<unsigned> currentVerticalSnapPointIndex,
        bool isMonitoringWheelEvents,
        std::optional<PlatformLayerIdentifier> scrollContainerLayer,
        std::optional<PlatformLayerIdentifier> scrolledContentsLayer,
        std::optional<PlatformLayerIdentifier> horizontalScrollbarLayer,
        std::optional<PlatformLayerIdentifier> verticalScrollbarLayer,
        bool mouseIsOverContentArea,
        MouseLocationState&&,
        ScrollbarHoverState&&,
        ScrollbarEnabledState&&,
        UserInterfaceLayoutDirection,
        ScrollbarWidth,
        bool useDarkAppearanceForScrollbars,
        RequestedKeyboardScrollData&&
    );
    ScrollingStateOverflowScrollingNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStateOverflowScrollingNode(const ScrollingStateOverflowScrollingNode&, ScrollingStateTree&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateOverflowScrollingNode, isOverflowScrollingNode())

#endif // ENABLE(ASYNC_SCROLLING)
