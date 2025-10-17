/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

#include "EventTrackingRegions.h"
#include "ScrollTypes.h"
#include "ScrollbarThemeComposite.h"
#include "ScrollingCoordinator.h"
#include "ScrollingStateScrollingNode.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class Scrollbar;

class ScrollingStateFrameScrollingNode final : public ScrollingStateScrollingNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateFrameScrollingNode, WEBCORE_EXPORT);
public:
    template<typename... Args> static Ref<ScrollingStateFrameScrollingNode> create(Args&&... args) { return adoptRef(*new ScrollingStateFrameScrollingNode(std::forward<Args>(args)...)); }

    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    virtual ~ScrollingStateFrameScrollingNode();

    float frameScaleFactor() const { return m_frameScaleFactor; }
    WEBCORE_EXPORT void setFrameScaleFactor(float);

    const EventTrackingRegions& eventTrackingRegions() const { return m_eventTrackingRegions; }
    WEBCORE_EXPORT void setEventTrackingRegions(const EventTrackingRegions&);

    ScrollBehaviorForFixedElements scrollBehaviorForFixedElements() const { return m_behaviorForFixed; }
    WEBCORE_EXPORT void setScrollBehaviorForFixedElements(ScrollBehaviorForFixedElements);

    FloatRect layoutViewport() const { return m_layoutViewport; };
    WEBCORE_EXPORT void setLayoutViewport(const FloatRect&);

    FloatPoint minLayoutViewportOrigin() const { return m_minLayoutViewportOrigin; }
    WEBCORE_EXPORT void setMinLayoutViewportOrigin(const FloatPoint&);

    FloatPoint maxLayoutViewportOrigin() const { return m_maxLayoutViewportOrigin; }
    WEBCORE_EXPORT void setMaxLayoutViewportOrigin(const FloatPoint&);

    std::optional<FloatSize> overrideVisualViewportSize() const { return m_overrideVisualViewportSize; };
    WEBCORE_EXPORT void setOverrideVisualViewportSize(std::optional<FloatSize>);

    int headerHeight() const { return m_headerHeight; }
    WEBCORE_EXPORT void setHeaderHeight(int);

    int footerHeight() const { return m_footerHeight; }
    WEBCORE_EXPORT void setFooterHeight(int);

    float topContentInset() const { return m_topContentInset; }
    WEBCORE_EXPORT void setTopContentInset(float);

    const LayerRepresentation& rootContentsLayer() const { return m_rootContentsLayer; }
    WEBCORE_EXPORT void setRootContentsLayer(const LayerRepresentation&);

    // This is a layer moved in the opposite direction to scrolling, for example for background-attachment:fixed
    const LayerRepresentation& counterScrollingLayer() const { return m_counterScrollingLayer; }
    WEBCORE_EXPORT void setCounterScrollingLayer(const LayerRepresentation&);

    // This is a clipping layer that will scroll with the page for all y-delta scroll values between 0
    // and topContentInset(). Once the y-deltas get beyond the content inset point, this layer no longer
    // needs to move. If the topContentInset() is 0, this layer does not need to move at all. This is
    // only used on the Mac.
    const LayerRepresentation& insetClipLayer() const { return m_insetClipLayer; }
    WEBCORE_EXPORT void setInsetClipLayer(const LayerRepresentation&);

    const LayerRepresentation& contentShadowLayer() const { return m_contentShadowLayer; }
    WEBCORE_EXPORT void setContentShadowLayer(const LayerRepresentation&);

    // The header and footer layers scroll vertically with the page, they should remain fixed when scrolling horizontally.
    const LayerRepresentation& headerLayer() const { return m_headerLayer; }
    WEBCORE_EXPORT void setHeaderLayer(const LayerRepresentation&);

    // The header and footer layers scroll vertically with the page, they should remain fixed when scrolling horizontally.
    const LayerRepresentation& footerLayer() const { return m_footerLayer; }
    WEBCORE_EXPORT void setFooterLayer(const LayerRepresentation&);

    // True when the visual viewport is smaller than the layout viewport, indicating that panning should be possible.
    bool visualViewportIsSmallerThanLayoutViewport() const { return m_visualViewportIsSmallerThanLayoutViewport; }
    WEBCORE_EXPORT void setVisualViewportIsSmallerThanLayoutViewport(bool);

    bool asyncFrameOrOverflowScrollingEnabled() const { return m_asyncFrameOrOverflowScrollingEnabled; }
    WEBCORE_EXPORT void setAsyncFrameOrOverflowScrollingEnabled(bool);

    bool scrollingPerformanceTestingEnabled() const { return m_scrollingPerformanceTestingEnabled; }
    WEBCORE_EXPORT void setScrollingPerformanceTestingEnabled(bool);

    bool wheelEventGesturesBecomeNonBlocking() const { return m_wheelEventGesturesBecomeNonBlocking; }
    WEBCORE_EXPORT void setWheelEventGesturesBecomeNonBlocking(bool);
    
    bool overlayScrollbarsEnabled() const { return m_overlayScrollbarsEnabled; }
    WEBCORE_EXPORT void setOverlayScrollbarsEnabled(bool);

    WEBCORE_EXPORT bool isMainFrame() const;
    
    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

private:
    WEBCORE_EXPORT ScrollingStateFrameScrollingNode(
        bool isMainFrame,
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
        OptionSet<SynchronousScrollingReason> synchronousScrollingReasons,
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
        RequestedKeyboardScrollData&&,
        float frameScaleFactor,
        EventTrackingRegions&&,
        std::optional<PlatformLayerIdentifier> rootContentsLayer,
        std::optional<PlatformLayerIdentifier> counterScrollingLayer,
        std::optional<PlatformLayerIdentifier> insetClipLayer,
        std::optional<PlatformLayerIdentifier> contentShadowLayer,
        int headerHeight,
        int footerHeight,
        ScrollBehaviorForFixedElements&&,
        float topContentInset,
        bool visualViewportIsSmallerThanLayoutViewport,
        bool asyncFrameOrOverflowScrollingEnabled,
        bool wheelEventGesturesBecomeNonBlocking,
        bool scrollingPerformanceTestingEnabled,
        FloatRect layoutViewport,
        FloatPoint minLayoutViewportOrigin,
        FloatPoint maxLayoutViewportOrigin,
        std::optional<FloatSize> overrideVisualViewportSize,
        bool overlayScrollbarsEnabled
    );

    ScrollingStateFrameScrollingNode(ScrollingStateTree&, ScrollingNodeType, ScrollingNodeID);
    ScrollingStateFrameScrollingNode(const ScrollingStateFrameScrollingNode&, ScrollingStateTree&);

    OptionSet<ScrollingStateNode::Property> applicableProperties() const final;

    LayerRepresentation m_rootContentsLayer;
    LayerRepresentation m_counterScrollingLayer;
    LayerRepresentation m_insetClipLayer;
    LayerRepresentation m_contentShadowLayer;
    LayerRepresentation m_headerLayer;
    LayerRepresentation m_footerLayer;

    EventTrackingRegions m_eventTrackingRegions;

    FloatRect m_layoutViewport;
    FloatPoint m_minLayoutViewportOrigin;
    FloatPoint m_maxLayoutViewportOrigin;
    std::optional<FloatSize> m_overrideVisualViewportSize;

    float m_frameScaleFactor { 1 };
    float m_topContentInset { 0 };
    int m_headerHeight { 0 };
    int m_footerHeight { 0 };
    ScrollBehaviorForFixedElements m_behaviorForFixed { ScrollBehaviorForFixedElements::StickToDocumentBounds };
    bool m_visualViewportIsSmallerThanLayoutViewport { false };
    bool m_asyncFrameOrOverflowScrollingEnabled { false };
    bool m_wheelEventGesturesBecomeNonBlocking { false };
    bool m_scrollingPerformanceTestingEnabled { false };
    bool m_overlayScrollbarsEnabled { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateFrameScrollingNode, isFrameScrollingNode())

#endif // ENABLE(ASYNC_SCROLLING)
