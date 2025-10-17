/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include "ScrollSnapOffsetsInfo.h"
#include "ScrollTypes.h"
#include "ScrollingCoordinator.h"
#include "ScrollingStateNode.h"

#if PLATFORM(COCOA)
OBJC_CLASS NSScrollerImp;
#endif

namespace WebCore {

struct ScrollbarHoverState {
    bool mouseIsOverHorizontalScrollbar { false };
    bool mouseIsOverVerticalScrollbar { false };

    friend bool operator==(const ScrollbarHoverState&, const ScrollbarHoverState&) = default;
};

struct MouseLocationState {
    IntPoint locationInHorizontalScrollbar;
    IntPoint locationInVerticalScrollbar;
};

struct ScrollbarEnabledState {
    bool horizontalScrollbarIsEnabled { false };
    bool verticalScrollbarIsEnabled { false };
};

class ScrollingStateScrollingNode : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateScrollingNode, WEBCORE_EXPORT);
public:
    virtual ~ScrollingStateScrollingNode();

    const FloatSize& scrollableAreaSize() const { return m_scrollableAreaSize; }
    WEBCORE_EXPORT void setScrollableAreaSize(const FloatSize&);

    const FloatSize& totalContentsSize() const { return m_totalContentsSize; }
    WEBCORE_EXPORT void setTotalContentsSize(const FloatSize&);

    const FloatSize& reachableContentsSize() const { return m_reachableContentsSize; }
    WEBCORE_EXPORT void setReachableContentsSize(const FloatSize&);

    const FloatPoint& scrollPosition() const { return m_scrollPosition; }
    WEBCORE_EXPORT void setScrollPosition(const FloatPoint&);

    const IntPoint& scrollOrigin() const { return m_scrollOrigin; }
    WEBCORE_EXPORT void setScrollOrigin(const IntPoint&);

    const FloatScrollSnapOffsetsInfo& snapOffsetsInfo() const { return m_snapOffsetsInfo; }
    WEBCORE_EXPORT void setSnapOffsetsInfo(const FloatScrollSnapOffsetsInfo& newOffsetsInfo);

    std::optional<unsigned> currentHorizontalSnapPointIndex() const { return m_currentHorizontalSnapPointIndex; }
    WEBCORE_EXPORT void setCurrentHorizontalSnapPointIndex(std::optional<unsigned>);

    std::optional<unsigned> currentVerticalSnapPointIndex() const { return m_currentVerticalSnapPointIndex; }
    WEBCORE_EXPORT void setCurrentVerticalSnapPointIndex(std::optional<unsigned>);

    const ScrollableAreaParameters& scrollableAreaParameters() const { return m_scrollableAreaParameters; }
    WEBCORE_EXPORT void setScrollableAreaParameters(const ScrollableAreaParameters& params);

#if ENABLE(SCROLLING_THREAD)
    OptionSet<SynchronousScrollingReason> synchronousScrollingReasons() const { return m_synchronousScrollingReasons; }
    WEBCORE_EXPORT void setSynchronousScrollingReasons(OptionSet<SynchronousScrollingReason>);
    bool hasSynchronousScrollingReasons() const { return !m_synchronousScrollingReasons.isEmpty(); }
#endif

    const RequestedKeyboardScrollData& keyboardScrollData() const { return m_keyboardScrollData; }
    WEBCORE_EXPORT void setKeyboardScrollData(const RequestedKeyboardScrollData&);

    const RequestedScrollData& requestedScrollData() const { return m_requestedScrollData; }

    enum class CanMergeScrollData : bool { No, Yes };
    WEBCORE_EXPORT void setRequestedScrollData(RequestedScrollData&&, CanMergeScrollData = CanMergeScrollData::Yes);

    WEBCORE_EXPORT bool hasScrollPositionRequest() const;

    bool isMonitoringWheelEvents() const { return m_isMonitoringWheelEvents; }
    WEBCORE_EXPORT void setIsMonitoringWheelEvents(bool);

    const LayerRepresentation& scrollContainerLayer() const { return m_scrollContainerLayer; }
    WEBCORE_EXPORT void setScrollContainerLayer(const LayerRepresentation&);

    // This is a layer with the contents that move.
    const LayerRepresentation& scrolledContentsLayer() const { return m_scrolledContentsLayer; }
    WEBCORE_EXPORT void setScrolledContentsLayer(const LayerRepresentation&);

    const LayerRepresentation& horizontalScrollbarLayer() const { return m_horizontalScrollbarLayer; }
    WEBCORE_EXPORT void setHorizontalScrollbarLayer(const LayerRepresentation&);

    const LayerRepresentation& verticalScrollbarLayer() const { return m_verticalScrollbarLayer; }
    WEBCORE_EXPORT void setVerticalScrollbarLayer(const LayerRepresentation&);

#if PLATFORM(MAC)
    NSScrollerImp *verticalScrollerImp() const { return m_verticalScrollerImp.get(); }
    NSScrollerImp *horizontalScrollerImp() const { return m_horizontalScrollerImp.get(); }
#endif
    ScrollbarHoverState scrollbarHoverState() const { return m_scrollbarHoverState; }
    WEBCORE_EXPORT void setScrollbarHoverState(ScrollbarHoverState);

    ScrollbarEnabledState scrollbarEnabledState() const { return m_scrollbarEnabledState; }
    WEBCORE_EXPORT void setScrollbarEnabledState(ScrollbarOrientation, bool);

    void setScrollerImpsFromScrollbars(Scrollbar* verticalScrollbar, Scrollbar* horizontalScrollbar);

    WEBCORE_EXPORT void setMouseIsOverContentArea(bool);
    bool mouseIsOverContentArea() const { return m_mouseIsOverContentArea; }

    WEBCORE_EXPORT void setMouseMovedInContentArea(const MouseLocationState&);
    const MouseLocationState& mouseLocationState() const { return m_mouseLocationState; }

    WEBCORE_EXPORT void setScrollbarLayoutDirection(UserInterfaceLayoutDirection);
    UserInterfaceLayoutDirection scrollbarLayoutDirection() const { return m_scrollbarLayoutDirection; }

    WEBCORE_EXPORT void setScrollbarWidth(ScrollbarWidth);
    ScrollbarWidth scrollbarWidth() const { return m_scrollbarWidth; }

    WEBCORE_EXPORT void setUseDarkAppearanceForScrollbars(bool);
    bool useDarkAppearanceForScrollbars() const { return m_useDarkAppearanceForScrollbars; }

protected:
    ScrollingStateScrollingNode(
        ScrollingNodeType,
        ScrollingNodeID,
        Vector<Ref<ScrollingStateNode>>&&,
        OptionSet<ScrollingStateNodeProperty>,
        std::optional<PlatformLayerIdentifier>,
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
    ScrollingStateScrollingNode(ScrollingStateTree&, ScrollingNodeType, ScrollingNodeID);
    ScrollingStateScrollingNode(const ScrollingStateScrollingNode&, ScrollingStateTree&);

    OptionSet<Property> applicableProperties() const override;
    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

private:
    FloatSize m_scrollableAreaSize;
    FloatSize m_totalContentsSize;
    FloatSize m_reachableContentsSize;
    FloatPoint m_scrollPosition;
    IntPoint m_scrollOrigin;

    FloatScrollSnapOffsetsInfo m_snapOffsetsInfo;
    std::optional<unsigned> m_currentHorizontalSnapPointIndex;
    std::optional<unsigned> m_currentVerticalSnapPointIndex;

    LayerRepresentation m_scrollContainerLayer;
    LayerRepresentation m_scrolledContentsLayer;
    LayerRepresentation m_horizontalScrollbarLayer;
    LayerRepresentation m_verticalScrollbarLayer;
    
    ScrollbarHoverState m_scrollbarHoverState;
    MouseLocationState m_mouseLocationState;
    ScrollbarEnabledState m_scrollbarEnabledState;

#if PLATFORM(MAC)
    RetainPtr<NSScrollerImp> m_verticalScrollerImp;
    RetainPtr<NSScrollerImp> m_horizontalScrollerImp;
#endif

    ScrollableAreaParameters m_scrollableAreaParameters;
    RequestedScrollData m_requestedScrollData;
    RequestedKeyboardScrollData m_keyboardScrollData;
#if ENABLE(SCROLLING_THREAD)
    OptionSet<SynchronousScrollingReason> m_synchronousScrollingReasons;
#endif
    UserInterfaceLayoutDirection m_scrollbarLayoutDirection { UserInterfaceLayoutDirection::LTR };
    ScrollbarWidth m_scrollbarWidth { ScrollbarWidth::Auto };

    bool m_useDarkAppearanceForScrollbars { false };
    bool m_isMonitoringWheelEvents { false };
    bool m_mouseIsOverContentArea { false };

};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateScrollingNode, isScrollingNode())

#endif // ENABLE(ASYNC_SCROLLING)
