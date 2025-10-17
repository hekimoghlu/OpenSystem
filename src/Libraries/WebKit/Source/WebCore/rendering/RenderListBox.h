/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#include "RenderBlockFlow.h"
#include "ScrollableArea.h"

namespace WebCore {

class HTMLOptGroupElement;
class HTMLOptionElement;
class HTMLSelectElement;

class RenderListBox final : public RenderBlockFlow, public ScrollableArea {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderListBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderListBox);
public:
    RenderListBox(HTMLSelectElement&, RenderStyle&&);
    virtual ~RenderListBox();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    HTMLSelectElement& selectElement() const;

    void selectionChanged();

    void setOptionsChanged(bool changed) { m_optionsChanged = changed; }

    int listIndexAtOffset(const LayoutSize&) const;
    LayoutRect itemBoundingBoxRect(const LayoutPoint&, int index) const;

    std::optional<LayoutRect> localBoundsOfOption(const HTMLOptionElement&) const;
    std::optional<LayoutRect> localBoundsOfOptGroup(const HTMLOptGroupElement&) const;

    bool scrollToRevealElementAtListIndex(int index);
    bool listIndexIsVisible(int index);

    int scrollToward(const IntPoint&); // Returns the new index or -1 if no scroll occurred

    unsigned size() const;

    bool scroll(ScrollDirection, ScrollGranularity, unsigned stepCount = 1, Element** stopElement = nullptr, RenderBox* startBox = nullptr, const IntPoint& wheelEventAbsolutePoint = IntPoint()) override;
    std::optional<FrameIdentifier> rootFrameID() const final;

private:
    bool isVisibleToHitTesting() const final;

    void willBeDestroyed() override;

    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderListBox"_s; }

    void updateFromElement() override;
    bool hasControlClip() const override { return true; }
    void paintObject(PaintInfo&, const LayoutPoint&) override;
    LayoutRect controlClipRect(const LayoutPoint&) const override;

    bool isPointInOverflowControl(HitTestResult&, const LayoutPoint& locationInContainer, const LayoutPoint& accumulatedOffset) override;

    bool logicalScroll(ScrollLogicalDirection, ScrollGranularity, unsigned stepCount = 1, Element** stopElement = nullptr) override;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const override;
    void computePreferredLogicalWidths() override;
    LayoutUnit baselinePosition(FontBaseline, bool firstLine, LineDirectionMode, LinePositionMode = PositionOnContainingLine) const override;
    LogicalExtentComputedValues computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const override;

    void layout() override;

    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint& additionalOffset, const RenderLayerModelObject* paintContainer = nullptr) const override;

    bool canBeProgramaticallyScrolled() const override { return true; }
    void autoscroll(const IntPoint&) override;
    void stopAutoscroll() override;

    virtual bool shouldPanScroll() const { return true; }
    void panScroll(const IntPoint&) override;

    int verticalScrollbarWidth() const override;
    int horizontalScrollbarHeight() const override;
    int scrollLeft() const override;
    int scrollTop() const override;
    int scrollWidth() const override;
    int scrollHeight() const override;
    void setScrollLeft(int, const ScrollPositionChangeOptions&) override;
    void setScrollTop(int, const ScrollPositionChangeOptions&) override;

    int logicalScrollTop() const;
    void setLogicalScrollTop(int);

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    // ScrollableArea interface.
    bool hasSteppedScrolling() const final { return true; }

    void setScrollOffset(const ScrollOffset&) final;

    ScrollPosition scrollPosition() const final;
    ScrollPosition minimumScrollPosition() const final;
    ScrollPosition maximumScrollPosition() const final;

    void invalidateScrollbarRect(Scrollbar&, const IntRect&) final;
    bool isActive() const final;
    bool isScrollCornerVisible() const final { return false; } // We don't support resize on list boxes yet. If we did these would have to change.
    IntRect scrollCornerRect() const final { return IntRect(); }
    void invalidateScrollCornerRect(const IntRect&) final { }
    IntRect convertFromScrollbarToContainingView(const Scrollbar&, const IntRect&) const final;
    IntRect convertFromContainingViewToScrollbar(const Scrollbar&, const IntRect&) const final;
    IntPoint convertFromScrollbarToContainingView(const Scrollbar&, const IntPoint&) const final;
    IntPoint convertFromContainingViewToScrollbar(const Scrollbar&, const IntPoint&) const final;
    Scrollbar* verticalScrollbar() const final;
    Scrollbar* horizontalScrollbar() const final;
    IntSize contentsSize() const final;
    IntSize visibleSize() const final { return IntSize(width(), height()); }
    IntPoint lastKnownMousePositionInView() const final;
    bool isHandlingWheelEvent() const final;
    bool shouldSuspendScrollAnimations() const final;
    bool forceUpdateScrollbarsOnMainThreadForPerformanceTesting() const final;

    ScrollableArea* enclosingScrollableArea() const final;
    bool isScrollableOrRubberbandable() final;
    bool hasScrollableOrRubberbandableAncestor() final;
    IntRect scrollableAreaBoundingBox(bool* = nullptr) const final;
    bool mockScrollbarsControllerEnabled() const final;
    void logMockScrollbarsControllerMessage(const String&) const final;
    String debugDescription() const final;
    void didStartScrollAnimation() final;

    bool useDarkAppearance() const final;

    // NOTE: This should only be called by the overridden setScrollOffset from ScrollableArea.
    void scrollTo(const ScrollPosition&);

    void scrollToPosition(int positionIndex);

    using PaintFunction = Function<void(PaintInfo&, const LayoutPoint&, int listItemIndex)>;
    void paintItem(PaintInfo&, const LayoutPoint&, const PaintFunction&);

    void setHasScrollbar(ScrollbarOrientation);
    Ref<Scrollbar> createScrollbar(ScrollbarOrientation);
    void destroyScrollbar();

    int maximumNumberOfItemsThatFitInPaddingAfterArea() const;

    int numberOfVisibleItemsInPaddingBefore() const;
    int numberOfVisibleItemsInPaddingAfter() const;

    void computeFirstIndexesVisibleInPaddingBeforeAfterAreas();

    LayoutUnit itemLogicalHeight() const;

    enum class ConsiderPadding : bool { No, Yes };
    int numVisibleItems(ConsiderPadding = ConsiderPadding::No) const;
    int numItems() const;
    LayoutUnit listLogicalHeight() const;

    std::optional<int> optionRowIndex(const HTMLOptionElement&) const;

    float deviceScaleFactor() const final;

    LayoutRect rectForScrollbar(const Scrollbar&) const;

    void paintScrollbar(PaintInfo&, const LayoutPoint&, Scrollbar&);
    void paintItemForeground(PaintInfo&, const LayoutPoint&, int listIndex);
    void paintItemBackground(PaintInfo&, const LayoutPoint&, int listIndex);
    void scrollToRevealSelection();

    ScrollbarOrientation scrollbarOrientationForWritingMode() const;

    bool shouldPlaceVerticalScrollbarOnLeft() const final { return RenderBlockFlow::shouldPlaceVerticalScrollbarOnLeft(); }

    int indexOffset() const;

    bool m_optionsChanged { true };
    bool m_scrollToRevealSelectionAfterLayout { false };
    bool m_inAutoscroll { false };
    int m_optionsLogicalWidth { 0 };

    RefPtr<Scrollbar> m_scrollbar;

    // Note: This is based on item index rather than a pixel offset.
    ScrollPosition m_scrollPosition;

    std::optional<int> m_indexOfFirstVisibleItemInsidePaddingBeforeArea;
    std::optional<int> m_indexOfFirstVisibleItemInsidePaddingAfterArea;

};

} // namepace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RenderListBox)
    static bool isType(const WebCore::RenderObject& renderer) { return renderer.isRenderListBox(); }
    static bool isType(const WebCore::ScrollableArea& area) { return area.isListBox(); }
SPECIALIZE_TYPE_TRAITS_END()
