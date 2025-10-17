/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

#include "RenderLayer.h"
#include "ScrollableArea.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderMarquee;

class RenderLayerScrollableArea final : public ScrollableArea, public CanMakeCheckedPtr<RenderLayerScrollableArea> {
    WTF_MAKE_TZONE_ALLOCATED(RenderLayerScrollableArea);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderLayerScrollableArea);
public:
    explicit RenderLayerScrollableArea(RenderLayer&);
    virtual ~RenderLayerScrollableArea();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    RenderLayer& layer() { return m_layer; }

    void clear();

    RenderMarquee* marquee() const { return m_marquee.get(); }
    void updateMarqueePosition();
    void createOrDestroyMarquee();

    void restoreScrollPosition();

#if ENABLE(IOS_TOUCH_EVENTS)
    void registerAsTouchEventListenerForScrolling();
    void unregisterAsTouchEventListenerForScrolling();
#endif

    void setPostLayoutScrollPosition(std::optional<ScrollPosition>);
    void applyPostLayoutScrollPositionIfNeeded();
    bool hasPostLayoutScrollPosition() { return !!m_postLayoutScrollPosition; }

    int scrollWidth() const;
    int scrollHeight() const;

    void panScrollFromPoint(const IntPoint&);

    // Scrolling methods for layers that can scroll their overflow.
    WEBCORE_EXPORT void scrollByRecursively(const IntSize& delta, ScrollableArea** scrolledArea = nullptr);

    // Attempt to scroll the given ScrollOffset, returning the real target offset after it has
    // been adjusted by scroll snapping.
    WEBCORE_EXPORT ScrollOffset scrollToOffset(const ScrollOffset&, const ScrollPositionChangeOptions& = ScrollPositionChangeOptions::createProgrammatic());

    void scrollToXPosition(int x, const ScrollPositionChangeOptions&);
    void scrollToYPosition(int y, const ScrollPositionChangeOptions&);
    void setScrollPosition(const ScrollPosition&, const ScrollPositionChangeOptions&);

    // These are only used by marquee.
    void scrollToXOffset(int x) { scrollToOffset(ScrollOffset(x, scrollOffset().y()), ScrollPositionChangeOptions::createProgrammaticUnclamped()); }
    void scrollToYOffset(int y) { scrollToOffset(ScrollOffset(scrollOffset().x(), y), ScrollPositionChangeOptions::createProgrammaticUnclamped()); }

    bool scrollsOverflow() const;
    bool hasScrollableHorizontalOverflow() const;
    bool hasScrollableVerticalOverflow() const;
    bool hasScrollbars() const { return horizontalScrollbar() || verticalScrollbar(); }
    bool hasHorizontalScrollbar() const { return horizontalScrollbar(); }
    bool hasVerticalScrollbar() const { return verticalScrollbar(); }
    void setHasHorizontalScrollbar(bool);
    void setHasVerticalScrollbar(bool);
    
    bool needsAnimatedScroll() const final { return m_isRegisteredForAnimatedScroll; }
    
    OverscrollBehavior horizontalOverscrollBehavior() const final;
    OverscrollBehavior verticalOverscrollBehavior() const final;

    Color scrollbarThumbColorStyle() const final;
    Color scrollbarTrackColorStyle() const final;
    ScrollbarGutter scrollbarGutterStyle() const final;
    ScrollbarWidth scrollbarWidthStyle() const final;

    bool requiresScrollPositionReconciliation() const { return m_requiresScrollPositionReconciliation; }
    void setRequiresScrollPositionReconciliation(bool requiresReconciliation = true) { m_requiresScrollPositionReconciliation = requiresReconciliation; }

    // Returns true when the layer could do touch scrolling, but doesn't look at whether there is actually scrollable overflow.
    bool canUseCompositedScrolling() const;
    // Returns true when there is actually scrollable overflow (requires layout to be up-to-date).
    bool hasCompositedScrollableOverflow() const { return m_hasCompositedScrollableOverflow; }

    int verticalScrollbarWidth(OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, bool isHorizontalWritingMode = true) const;
    int horizontalScrollbarHeight(OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, bool isHorizontalWritingMode = true) const;

    bool hasOverflowControls() const;
    bool hitTestOverflowControls(HitTestResult&, const IntPoint& localPoint);
    bool hitTestResizerInFragments(const LayerFragments&, const HitTestLocation&, LayoutPoint& pointInFragment) const;

    void paintOverflowControls(GraphicsContext&, const IntPoint&, const IntRect& damageRect, bool paintingOverlayControls = false);
    void paintScrollCorner(GraphicsContext&, const IntPoint&, const IntRect& damageRect);
    void paintResizer(GraphicsContext&, const LayoutPoint&, const LayoutRect& damageRect);
    void paintOverlayScrollbars(GraphicsContext&, const LayoutRect& damageRect, OptionSet<PaintBehavior>, RenderObject* subtreePaintRoot = nullptr);

    void updateScrollInfoAfterLayout();
    void updateScrollbarSteps();

    bool scroll(ScrollDirection, ScrollGranularity, unsigned stepCount = 1);

public:
    // All methods in this section override ScrollableaArea methods (final).
    void availableContentSizeChanged(AvailableSizeChangeReason) final;

    NativeScrollbarVisibility horizontalNativeScrollbarVisibility() const final;
    NativeScrollbarVisibility verticalNativeScrollbarVisibility() const final;

    bool canShowNonOverlayScrollbars() const final;

    ScrollPosition scrollPosition() const final { return m_scrollPosition; }

    Scrollbar* horizontalScrollbar() const final { return m_hBar.get(); }
    Scrollbar* verticalScrollbar() const final { return m_vBar.get(); }
    ScrollableArea* enclosingScrollableArea() const final;

    bool handleWheelEventForScrolling(const PlatformWheelEvent&, std::optional<WheelScrollGestureState>) final;
    bool isScrollableOrRubberbandable() final;
    bool hasScrollableOrRubberbandableAncestor() final;
    bool useDarkAppearance() const final;
    void updateSnapOffsets() final;

#if PLATFORM(IOS_FAMILY)
#if ENABLE(IOS_TOUCH_EVENTS)
    bool handleTouchEvent(const PlatformTouchEvent&) final;
#endif

    void didStartScroll() final;
    void didEndScroll() final;
    void didUpdateScroll() final;
#endif

    GraphicsLayer* layerForHorizontalScrollbar() const final;
    GraphicsLayer* layerForVerticalScrollbar() const final;
    GraphicsLayer* layerForScrollCorner() const final;

    bool usesCompositedScrolling() const final;
    bool usesAsyncScrolling() const final;

    bool shouldPlaceVerticalScrollbarOnLeft() const final;

    bool isRenderLayer() const final { return true; }
    void invalidateScrollbarRect(Scrollbar&, const IntRect&) final;
    void invalidateScrollCornerRect(const IntRect&) final;
    bool isActive() const final;
    bool isScrollCornerVisible() const final;
    IntRect scrollCornerRect() const final;
    IntRect convertFromScrollbarToContainingView(const Scrollbar&, const IntRect&) const final;
    IntRect convertFromContainingViewToScrollbar(const Scrollbar&, const IntRect&) const final;
    IntPoint convertFromScrollbarToContainingView(const Scrollbar&, const IntPoint&) const final;
    IntPoint convertFromContainingViewToScrollbar(const Scrollbar&, const IntPoint&) const final;
    void setScrollOffset(const ScrollOffset&) final;
    WEBCORE_EXPORT std::optional<ScrollingNodeID> scrollingNodeID() const final;

    IntRect visibleContentRectInternal(VisibleContentRectIncludesScrollbars, VisibleContentRectBehavior) const final;
    IntSize overhangAmount() const final;
    IntPoint lastKnownMousePositionInView() const final;
    bool isHandlingWheelEvent() const final;
    bool shouldSuspendScrollAnimations() const final;
    IntRect scrollableAreaBoundingBox(bool* isInsideFixed = nullptr) const final;
    bool isUserScrollInProgress() const final;
    bool isRubberBandInProgress() const final;
    bool forceUpdateScrollbarsOnMainThreadForPerformanceTesting() const final;
    bool isScrollSnapInProgress() const final;
    bool scrollAnimatorEnabled() const final;
    bool mockScrollbarsControllerEnabled() const final;
    void logMockScrollbarsControllerMessage(const String&) const final;

    String debugDescription() const final;
    void didStartScrollAnimation() final;

    IntSize visibleSize() const final;
    IntSize contentsSize() const final;
    IntSize reachableTotalContentsSize() const final;

    bool requestStartKeyboardScrollAnimation(const KeyboardScroll&) final;
    bool requestStopKeyboardScrollAnimation(bool immediate) final;

    bool requestScrollToPosition(const ScrollPosition&, const ScrollPositionChangeOptions& options) final;
    void stopAsyncAnimatedScroll() final;

    bool containsDirtyOverlayScrollbars() const { return m_containsDirtyOverlayScrollbars; }
    void setContainsDirtyOverlayScrollbars(bool dirtyScrollbars) { m_containsDirtyOverlayScrollbars = dirtyScrollbars; }

    void updateScrollbarsAfterStyleChange(const RenderStyle* oldStyle);
    void updateScrollbarsAfterLayout();

    bool positionOverflowControls(const IntSize&);

    void updateAllScrollbarRelatedStyle();

    LayoutUnit overflowTop() const;
    LayoutUnit overflowBottom() const;
    LayoutUnit overflowLeft() const;
    LayoutUnit overflowRight() const;

    RenderLayer::OverflowControlRects overflowControlsRects() const;

    bool overflowControlsIntersectRect(const IntRect& localRect) const;

    bool scrollingMayRevealBackground() const;

    void computeHasCompositedScrollableOverflow(LayoutUpToDate);

    // NOTE: This should only be called by the overridden setScrollOffset from ScrollableArea.
    void scrollTo(const ScrollPosition&);
    void updateCompositingLayersAfterScroll();

    IntSize scrollbarOffset(const Scrollbar&) const;

    std::optional<LayoutRect> updateScrollPosition(const ScrollPositionChangeOptions&, const LayoutRect& revealRect, const LayoutRect& localExposeRect);
    bool isVisibleToHitTesting() const final;
    void animatedScrollDidEnd() final;
    LayoutRect scrollRectToVisible(const LayoutRect& absoluteRect, const ScrollRectToVisibleOptions&);
    std::optional<LayoutRect> updateScrollPositionForScrollIntoView(const ScrollPositionChangeOptions&, const LayoutRect& revealRect, const LayoutRect& localExposeRect);
    void updateScrollAnchoringElement() final;
    void updateScrollPositionForScrollAnchoringController() final;
    void invalidateScrollAnchoringElement() final;
    ScrollAnchoringController* scrollAnchoringController() { return m_scrollAnchoringController.get(); }

    void createScrollbarsController() final;

    std::optional<FrameIdentifier> rootFrameID() const final;

    void scrollbarWidthChanged(ScrollbarWidth) override;

private:
    bool hasHorizontalOverflow() const;
    bool hasVerticalOverflow() const;

    bool showsOverflowControls() const;

    ScrollOffset clampScrollOffset(const ScrollOffset&) const;

    void computeScrollDimensions();
    void computeScrollOrigin();

    void updateScrollableAreaSet(bool hasOverflow);

    void updateScrollCornerStyle();
    void updateResizerStyle();

    void drawPlatformResizerImage(GraphicsContext&, const LayoutRect& resizerCornerRect);

    Ref<Scrollbar> createScrollbar(ScrollbarOrientation);
    void destroyScrollbar(ScrollbarOrientation);

    void clearScrollCorner();
    void clearResizer();

    void updateScrollbarPresenceAndState(std::optional<bool> hasHorizontalOverflow = std::nullopt, std::optional<bool> hasVerticalOverflow = std::nullopt);
    void registerScrollableAreaForAnimatedScroll();

    float deviceScaleFactor() const final;

private:
    bool m_scrollDimensionsDirty { true };
    bool m_inOverflowRelayout { false };
    bool m_registeredScrollableArea { false };
    bool m_hasCompositedScrollableOverflow { false };

#if PLATFORM(IOS_FAMILY) && ENABLE(IOS_TOUCH_EVENTS)
    bool m_registeredAsTouchEventListenerForScrolling { false };
#endif
    bool m_requiresScrollPositionReconciliation { false };
    bool m_containsDirtyOverlayScrollbars { false };
    bool m_updatingMarqueePosition { false };
    
    bool m_isRegisteredForAnimatedScroll { false };

    // The width/height of our scrolled area.
    int m_scrollWidth { 0 };
    int m_scrollHeight { 0 };

    RenderLayer& m_layer;
    ScrollPosition m_scrollPosition;
    std::optional<ScrollPosition> m_postLayoutScrollPosition;

    // For layers with overflow, we have a pair of scrollbars.
    RefPtr<Scrollbar> m_hBar;
    RefPtr<Scrollbar> m_vBar;

    IntPoint m_cachedOverlayScrollbarOffset;

    // Renderers to hold our custom scroll corner and resizer.
    RenderPtr<RenderScrollbarPart> m_scrollCorner;
    RenderPtr<RenderScrollbarPart> m_resizer;

    std::unique_ptr<RenderMarquee> m_marquee; // Used for <marquee>.

    std::unique_ptr<ScrollAnchoringController> m_scrollAnchoringController;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RenderLayerScrollableArea)
static bool isType(const WebCore::ScrollableArea& area) { return area.isRenderLayer(); }
SPECIALIZE_TYPE_TRAITS_END()
