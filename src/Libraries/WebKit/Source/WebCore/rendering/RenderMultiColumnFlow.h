/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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

#include "RenderFragmentedFlow.h"
#include <wtf/HashMap.h>

namespace WebCore {

class RenderMultiColumnSet;
class RenderMultiColumnSpannerPlaceholder;

class RenderMultiColumnFlow final : public RenderFragmentedFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMultiColumnFlow);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMultiColumnFlow);
public:
    RenderMultiColumnFlow(Document&, RenderStyle&&);
    virtual ~RenderMultiColumnFlow();

    RenderBlockFlow* multiColumnBlockFlow() const { return downcast<RenderBlockFlow>(parent()); }

    RenderMultiColumnSet* firstMultiColumnSet() const;
    RenderMultiColumnSet* lastMultiColumnSet() const;
    RenderBox* firstColumnSetOrSpanner() const;
    bool hasColumnSpanner() const { return !m_spannerMap.isEmpty(); }
    static RenderBox* nextColumnSetOrSpannerSiblingOf(const RenderBox*);
    static RenderBox* previousColumnSetOrSpannerSiblingOf(const RenderBox*);

    RenderMultiColumnSpannerPlaceholder* findColumnSpannerPlaceholder(const RenderBox* spanner) const;

    void layout() override;

    unsigned columnCount() const { return m_columnCount; }
    LayoutUnit columnWidth() const { return m_columnWidth; }
    LayoutUnit columnHeightAvailable() const { return m_columnHeightAvailable; }
    void setColumnHeightAvailable(LayoutUnit available) { m_columnHeightAvailable = available; }
    bool inBalancingPass() const { return m_inBalancingPass; }
    void setInBalancingPass(bool balancing) { m_inBalancingPass = balancing; }
    bool needsHeightsRecalculation() const { return m_needsHeightsRecalculation; }
    void setNeedsHeightsRecalculation(bool recalculate) { m_needsHeightsRecalculation = recalculate; }

    bool shouldRelayoutForPagination() const { return !m_inBalancingPass && m_needsHeightsRecalculation; }

    void setColumnCountAndWidth(unsigned count, LayoutUnit width)
    {
        m_columnCount = count;
        m_columnWidth = width;
    }

    bool progressionIsInline() const { return m_progressionIsInline; }
    void setProgressionIsInline(bool progressionIsInline) { m_progressionIsInline = progressionIsInline; }

    bool progressionIsReversed() const { return m_progressionIsReversed; }
    void setProgressionIsReversed(bool reversed) { m_progressionIsReversed = reversed; }

    RenderFragmentContainer* mapFromFlowToFragment(TransformState&) const final;

    // This method takes a logical offset and returns a physical translation that can be applied to map
    // a physical point (corresponding to the logical offset) into the fragment's physical coordinate space.
    LayoutSize physicalTranslationOffsetFromFlowToFragment(const RenderFragmentContainer*, const LayoutUnit) const;
    
    // The point is physical, and the result is a physical location within the fragment.
    RenderFragmentContainer* physicalTranslationFromFlowToFragment(LayoutPoint&) const;
    
    // This method is the inverse of the previous method and goes from fragment to flow.
    LayoutSize physicalTranslationFromFragmentToFlow(const RenderMultiColumnSet*, const LayoutPoint&) const;
    
    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;
    
    void mapAbsoluteToLocalPoint(OptionSet<MapCoordinatesMode>, TransformState&) const override;
    LayoutSize offsetFromContainer(RenderElement&, const LayoutPoint&, bool* offsetDependsOnPoint = nullptr) const override;
    
    // FIXME: Eventually as column and fragment flow threads start nesting, this will end up changing.
    bool shouldCheckColumnBreaks() const override;

    using SpannerMap = UncheckedKeyHashMap<SingleThreadWeakRef<const RenderBox>, SingleThreadWeakPtr<RenderMultiColumnSpannerPlaceholder>>;
    SpannerMap& spannerMap() { return m_spannerMap; }

private:
    ASCIILiteral renderName() const override;
    void addFragmentToThread(RenderFragmentContainer*) override;
    void willBeRemovedFromTree() override;
    void fragmentedFlowDescendantBoxLaidOut(RenderBox*) override;
    LogicalExtentComputedValues computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const override;
    LayoutUnit initialLogicalWidth() const override;
    void setPageBreak(const RenderBlock*, LayoutUnit offset, LayoutUnit spaceShortage) override;
    void updateMinimumPageHeight(const RenderBlock*, LayoutUnit offset, LayoutUnit minHeight) override;
    void updateSpaceShortageForSizeContainment(const RenderBlock*, LayoutUnit offset, LayoutUnit shortage) override;
    RenderFragmentContainer* fragmentAtBlockOffset(const RenderBox*, LayoutUnit, bool extendLastFragment = false) const override;
    void setFragmentRangeForBox(const RenderBox&, RenderFragmentContainer*, RenderFragmentContainer*) override;
    bool addForcedFragmentBreak(const RenderBlock*, LayoutUnit, RenderBox* breakChild, bool isBefore, LayoutUnit* offsetBreakAdjustment = 0) override;
    bool isPageLogicalHeightKnown() const override;

private:
    SpannerMap m_spannerMap;

    // The last set we worked on. It's not to be used as the "current set". The concept of a
    // "current set" is difficult, since layout may jump back and forth in the tree, due to wrong
    // top location estimates (due to e.g. margin collapsing), and possibly for other reasons.
    mutable SingleThreadWeakPtr<RenderMultiColumnSet> m_lastSetWorkedOn { nullptr };

    unsigned m_columnCount { 1 }; // The default column count/width that are based off our containing block width. These values represent only the default,
    LayoutUnit m_columnWidth { 0 }; // A multi-column block that is split across variable width pages or fragments will have different column counts and widths in each. These values will be cached (eventually) for multi-column blocks.

    LayoutUnit m_columnHeightAvailable; // Total height available to columns, or 0 if auto.
    bool m_inLayout { false }; // Set while we're laying out the flow thread, during which colum set heights are unknown.
    bool m_inBalancingPass { false }; // Guard to avoid re-entering column balancing.
    bool m_needsHeightsRecalculation { false };
    
    bool m_progressionIsInline { false };
    bool m_progressionIsReversed { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMultiColumnFlow, isRenderMultiColumnFlow())
