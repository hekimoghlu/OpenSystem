/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#include "InlineQuirks.h"

#include "InlineFormattingContext.h"
#include "InlineLineBox.h"
#include "LayoutBoxGeometry.h"
#include "RenderStyleInlines.h"

namespace WebCore {
namespace Layout {

InlineQuirks::InlineQuirks(const InlineFormattingContext& inlineFormattingContext)
    : m_inlineFormattingContext(inlineFormattingContext)
{
}

bool InlineQuirks::trailingNonBreakingSpaceNeedsAdjustment(bool isInIntrinsicWidthMode, bool lineHasOverflow) const
{
    if (isInIntrinsicWidthMode || !lineHasOverflow)
        return false;
    auto& rootStyle = formattingContext().root().style();
    return rootStyle.nbspMode() == NBSPMode::Space && rootStyle.textWrapMode() != TextWrapMode::NoWrap && rootStyle.whiteSpaceCollapse() != WhiteSpaceCollapse::BreakSpaces;
}

InlineLayoutUnit InlineQuirks::initialLineHeight() const
{
    ASSERT(!formattingContext().layoutState().inStandardsMode());
    return 0.f;
}

bool InlineQuirks::lineBreakBoxAffectsParentInlineBox(const LineBox& lineBox)
{
    // In quirks mode linebreak boxes (<br>) stop affecting the line box when (assume <br> is nested e.g. <span style="font-size: 100px"><br></span>)
    // 1. the root inline box has content <div>content<br>/div>
    // 2. there's at least one atomic inline level box on the line e.g <div><img><br></div> or <div><span><img></span><br></div>
    // 3. there's at least one inline box with content e.g. <div><span>content</span><br></div>
    if (lineBox.rootInlineBox().hasContent())
        return false;
    if (lineBox.hasAtomicInlineBox())
        return false;
    // At this point we either have only the <br> on the line or inline boxes with or without content.
    auto& inlineLevelBoxes = lineBox.nonRootInlineLevelBoxes();
    ASSERT(!inlineLevelBoxes.isEmpty());
    if (inlineLevelBoxes.size() == 1)
        return true;
    for (auto& inlineLevelBox : lineBox.nonRootInlineLevelBoxes()) {
        // Filter out empty inline boxes e.g. <div><span></span><span></span><br></div>
        if (inlineLevelBox.isInlineBox() && inlineLevelBox.hasContent())
            return false;
    }
    return true;
}

bool InlineQuirks::inlineBoxAffectsLineBox(const InlineLevelBox& inlineLevelBox) const
{
    ASSERT(!formattingContext().layoutState().inStandardsMode());
    ASSERT(inlineLevelBox.isInlineBox());
    // Inline boxes (e.g. root inline box or <span>) affects line boxes either through the strut or actual content.
    if (inlineLevelBox.hasContent())
        return true;
    if (inlineLevelBox.isRootInlineBox()) {
        // This root inline box has no direct text content and we are in non-standards mode.
        // Now according to legacy line layout, we need to apply the following list-item specific quirk:
        // We do not create markers for list items when the list-style-type is none, while other browsers do.
        // The side effect of having no marker is that in quirks mode we have to specifically check for list-item
        // and make sure it is treated as if it had content and stretched the line.
        // see LegacyInlineFlowBox c'tor.
        return inlineLevelBox.layoutBox().style().isOriginalDisplayListItemType();
    }
    // Non-root inline boxes (e.g. <span>).
    auto& boxGeometry = formattingContext().geometryForBox(inlineLevelBox.layoutBox());
    if (boxGeometry.horizontalBorderAndPadding()) {
        // Horizontal border and padding make the inline box stretch the line (e.g. <span style="padding: 10px;"></span>).
        return true;
    }
    return false;
}

std::optional<LayoutUnit> InlineQuirks::initialLetterAlignmentOffset(const Box& floatBox, const RenderStyle& lineBoxStyle) const
{
    ASSERT(floatBox.isFloatingPositioned());
    if (!floatBox.style().lineBoxContain().contains(LineBoxContain::InitialLetter))
        return { };
    auto& primaryFontMetrics = lineBoxStyle.fontCascade().metricsOfPrimaryFont();
    auto lineHeight = [&]() -> InlineLayoutUnit {
        if (lineBoxStyle.lineHeight().isNormal())
            return primaryFontMetrics.intAscent() + primaryFontMetrics.intDescent();
        return lineBoxStyle.computedLineHeight();
    };
    auto& floatBoxGeometry = formattingContext().geometryForBox(floatBox);
    return LayoutUnit { primaryFontMetrics.intAscent() + (lineHeight() - primaryFontMetrics.intHeight()) / 2 - primaryFontMetrics.intCapHeight() - floatBoxGeometry.marginBorderAndPaddingBefore() };
}

std::optional<InlineRect> InlineQuirks::adjustedRectForLineGridLineAlign(const InlineRect& rect) const
{
    auto& rootBoxStyle = formattingContext().root().style();
    auto& parentBlockLayoutState = formattingContext().layoutState().parentBlockLayoutState();

    if (rootBoxStyle.lineAlign() == LineAlign::None)
        return { };
    if (!parentBlockLayoutState.lineGrid())
        return { };

    // This implement the legacy -webkit-line-align property.
    // It snaps line edges to a grid defined by an ancestor box.
    auto& lineGrid = *parentBlockLayoutState.lineGrid();
    auto offset = InlineLayoutUnit { lineGrid.layoutOffset.width() + lineGrid.gridOffset.width() };
    auto columnWidth = lineGrid.columnWidth;
    auto leftShift = fmodf(columnWidth - fmodf(rect.left() + offset, columnWidth), columnWidth);
    auto rightShift = -fmodf(rect.right() + offset, columnWidth);

    auto adjustedRect = rect;
    adjustedRect.shiftLeftBy(leftShift);
    adjustedRect.shiftRightBy(rightShift);

    if (adjustedRect.isEmpty())
        return { };

    return adjustedRect;
}

std::optional<InlineLayoutUnit> InlineQuirks::adjustmentForLineGridLineSnap(const LineBox& lineBox) const
{
    auto& rootBoxStyle = formattingContext().root().style();
    auto& inlineLayoutState = formattingContext().layoutState();

    if (rootBoxStyle.lineSnap() == LineSnap::None)
        return { };
    if (!inlineLayoutState.parentBlockLayoutState().lineGrid())
        return { };

    // This implement the legacy -webkit-line-snap property.
    // It snaps line baselines to a grid defined by an ancestor box.

    auto& lineGrid = *inlineLayoutState.parentBlockLayoutState().lineGrid();

    auto gridLineHeight = lineGrid.rowHeight;
    if (!roundToInt(gridLineHeight))
        return { };

    auto& gridFontMetrics = lineGrid.primaryFont->fontMetrics();
    auto lineGridFontAscent = gridFontMetrics.intAscent(lineBox.baselineType());
    auto lineGridFontHeight = gridFontMetrics.intHeight();
    auto lineGridHalfLeading = (gridLineHeight - lineGridFontHeight) / 2;

    auto firstLineTop = lineGrid.topRowOffset + lineGrid.gridOffset.height();

    if (lineGrid.paginationOrigin && lineGrid.pageLogicalTop > firstLineTop)
        firstLineTop = lineGrid.paginationOrigin->height() + lineGrid.pageLogicalTop;

    auto firstTextTop = firstLineTop + lineGridHalfLeading;
    auto firstBaselinePosition = firstTextTop + lineGridFontAscent;

    auto rootInlineBoxTop = lineBox.logicalRect().top() + lineBox.logicalRectForRootInlineBox().top();

    auto ascent = lineBox.rootInlineBox().ascent();
    auto logicalHeight = ascent + lineBox.rootInlineBox().descent();
    auto currentBaselinePosition = rootInlineBoxTop + ascent + lineGrid.layoutOffset.height();

    if (rootBoxStyle.lineSnap() == LineSnap::Contain) {
        if (logicalHeight <= lineGridFontHeight)
            firstTextTop += (lineGridFontHeight - logicalHeight) / 2;
        else {
            LayoutUnit numberOfLinesWithLeading { ceilf(static_cast<float>(logicalHeight - lineGridFontHeight) / gridLineHeight) };
            LayoutUnit totalHeight = lineGridFontHeight + numberOfLinesWithLeading * gridLineHeight;
            firstTextTop += (totalHeight - logicalHeight) / 2;
        }
        firstBaselinePosition = firstTextTop + ascent;
    }

    // If we're above the first line, just push to the first line.
    if (currentBaselinePosition < firstBaselinePosition)
        return firstBaselinePosition - currentBaselinePosition;

    // Otherwise we're in the middle of the grid somewhere. Just push to the next line.
    auto baselineOffset = currentBaselinePosition - firstBaselinePosition;
    auto remainder = roundToInt(baselineOffset) % roundToInt(gridLineHeight);
    if (!remainder)
        return { };

    return gridLineHeight - remainder;
}

}
}

