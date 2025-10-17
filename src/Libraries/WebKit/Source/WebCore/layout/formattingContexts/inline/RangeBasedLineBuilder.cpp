/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "RangeBasedLineBuilder.h"

#include "InlineFormattingContext.h"

namespace WebCore {
namespace Layout {

RangeBasedLineBuilder::RangeBasedLineBuilder(InlineFormattingContext& inlineFormattingContext, HorizontalConstraints rootHorizontalConstraints, const InlineItemList& inlineItemList)
    : AbstractLineBuilder(inlineFormattingContext, inlineFormattingContext.root(), rootHorizontalConstraints, inlineItemList)
    , m_textOnlySimpleLineBuilder(inlineFormattingContext, downcast<ElementBox>(inlineItemList[0].layoutBox()), rootHorizontalConstraints, inlineItemList)
{
}

LineLayoutResult RangeBasedLineBuilder::layoutInlineContent(const LineInput& lineInput, const std::optional<PreviousLine>& previousLine)
{
    // 1. Shrink the layout range that we can run text-only builder on (currently it's just the opening/closing inline box)
    // 2. Run text-only line builder
    // 3. Insert/append the missing inline box run
    auto isFirstLine = !lineInput.needsLayoutRange.startIndex();

    auto adjustedNeedsLayoutRange = [&] {
        auto needsLayoutRange = lineInput.needsLayoutRange;
        if (isFirstLine) {
            ASSERT(m_inlineItemList.front().isInlineBoxStart());
            ASSERT(!needsLayoutRange.start.offset);
            // Skip leading InlineItemStart (e.g. <span>)
            ++needsLayoutRange.start.index;
        }
        // SKip trailing InlineItemEnd (e.g. </span>)
        ASSERT(m_inlineItemList.back().isInlineBoxEnd());
        ASSERT(!needsLayoutRange.end.offset);
        --needsLayoutRange.end.index;
        return needsLayoutRange;
    };
    auto needsLayoutRange = adjustedNeedsLayoutRange();
    ASSERT(!needsLayoutRange.isEmpty());

    auto lineLayoutResult = m_textOnlySimpleLineBuilder.layoutInlineContent({ needsLayoutRange, lineInput.initialLogicalRect }, previousLine);

    auto insertLeadingInlineBoxRun = [&] {
        auto& leadingInlineItem = m_inlineItemList.front();
        ASSERT(leadingInlineItem.isInlineBoxStart());

        if (isFirstLine) {
            ASSERT(!previousLine);
            lineLayoutResult.inlineContent.insert(0, { leadingInlineItem, leadingInlineItem.firstLineStyle(), { } });
            lineLayoutResult.inlineItemRange.start = lineInput.needsLayoutRange.start;
            return;
        }
        // Subsequent lines need leading spanning inline box run.
        lineLayoutResult.inlineContent.insert(0, { leadingInlineItem, { }, { } });
    };
    insertLeadingInlineBoxRun();

    auto appendTrailingInlineBoxRunIfNeeded = [&] {
        if (lineLayoutResult.inlineItemRange.end != needsLayoutRange.end)
            return;
        auto& trailingInlineItem = m_inlineItemList.back();
        lineLayoutResult.inlineContent.append({ trailingInlineItem, isFirstLine ? trailingInlineItem.firstLineStyle() : trailingInlineItem.style(), lineLayoutResult.contentGeometry.logicalWidth });
        lineLayoutResult.inlineItemRange.end = lineInput.needsLayoutRange.end;
    };
    appendTrailingInlineBoxRunIfNeeded();

    return lineLayoutResult;
}

bool RangeBasedLineBuilder::isEligibleForRangeInlineLayout(const InlineFormattingContext& inlineFormattingContext, const InlineContentCache::InlineItems& inlineItems, const PlacedFloats& placedFloats)
{
    if (inlineItems.isEmpty())
        return false;
    // Range based line builder only supports the following content <inline box>eligible for text only layout</inline box>
    auto& inlineItemList = inlineItems.content();
    auto isFullyNestedContent = inlineItems.inlineBoxCount() == 1 && inlineItemList.first().isInlineBoxStart() && inlineItemList.last().isInlineBoxEnd() && inlineItemList.size() > 2;
    if (!isFullyNestedContent)
        return false;

    auto& inlineBox = inlineItemList.first().layoutBox();
    auto& inlineBoxGeometry = inlineFormattingContext.geometryForBox(inlineBox);
    if (inlineBoxGeometry.horizontalMarginBorderAndPadding()) {
        // FIXME: Add start decoration support is just a matter of shrinking the available space for the first line (or on subsequent lines when decoration break is present)
        return false;
    }
    if (inlineBox.style().boxDecorationBreak() != RenderStyle::initialBoxDecorationBreak())
        return false;

    if (inlineFormattingContext.layoutState().parentBlockLayoutState().lineClamp())
        return false;

    // Check the nested text content.
    if (!inlineItems.hasTextAndLineBreakOnlyContent() || inlineItems.requiresVisualReordering() || !placedFloats.isEmpty() || inlineItems.hasTextAutospace())
        return false;

    if (!TextOnlySimpleLineBuilder::isEligibleForSimplifiedInlineLayoutByStyle(inlineFormattingContext.root().style()) || !TextOnlySimpleLineBuilder::isEligibleForSimplifiedInlineLayoutByStyle(inlineBox.style()))
        return false;

    return true;
}

}
}

