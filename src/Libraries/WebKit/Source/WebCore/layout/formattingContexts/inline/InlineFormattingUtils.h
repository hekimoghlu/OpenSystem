/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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

#include "InlineLineBuilder.h"

namespace WebCore {
namespace Layout {

class FloatingContext;
class InlineFormattingContext;
class InlineLevelBox;

class InlineFormattingUtils {
public:
    InlineFormattingUtils(const InlineFormattingContext&);

    InlineLayoutUnit logicalTopForNextLine(const LineLayoutResult&, const InlineRect& lineLogicalRect, const FloatingContext&) const;

    ContentHeightAndMargin inlineBlockContentHeightAndMargin(const Box&, const HorizontalConstraints&, const OverriddenVerticalValues&) const;
    ContentWidthAndMargin inlineBlockContentWidthAndMargin(const Box&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    enum class IsIntrinsicWidthMode : bool { No, Yes };
    InlineLayoutUnit computedTextIndent(IsIntrinsicWidthMode, std::optional<bool> previousLineEndsWithLineBreak, InlineLayoutUnit availableWidth) const;

    bool inlineLevelBoxAffectsLineBox(const InlineLevelBox&) const;

    InlineLayoutUnit initialLineHeight(bool isFirstLine) const;

    FloatingContext::Constraints floatConstraintsForLine(InlineLayoutUnit lineLogicalTop, InlineLayoutUnit contentLogicalHeight, const FloatingContext&) const;

    static InlineRect flipVisualRectToLogicalForWritingMode(const InlineRect& visualRect, WritingMode);

    void adjustMarginStartForListMarker(const ElementBox&, LayoutUnit nestedListMarkerMarginStart, InlineLayoutUnit rootInlineBoxOffset) const;

    static InlineLayoutUnit horizontalAlignmentOffset(const RenderStyle& rootStyle, InlineLayoutUnit contentLogicalRight, InlineLayoutUnit lineLogicalRight, InlineLayoutUnit hangingTrailingWidth, const Line::RunList& runs, bool isLastLine, std::optional<TextDirection> inlineBaseDirectionOverride = std::nullopt);

    static InlineItemPosition leadingInlineItemPositionForNextLine(InlineItemPosition lineContentEnd, std::optional<InlineItemPosition> previousLineContentEnd, bool lineHasIntrusiveOrNewlyPlacedFloat, InlineItemPosition layoutRangeEnd);

    InlineLayoutUnit inlineItemWidth(const InlineItem&, InlineLayoutUnit contentLogicalLeft, bool useFirstLineStyle) const;

    size_t nextWrapOpportunity(size_t startIndex, const InlineItemRange& layoutRange, std::span<const InlineItem>) const;

    static std::pair<InlineLayoutUnit, InlineLayoutUnit> textEmphasisForInlineBox(const Box&, const ElementBox& rootBox);

    static LineEndingTruncationPolicy lineEndingTruncationPolicy(const RenderStyle& rootStyle, size_t numberOfContentfulLines, std::optional<size_t> numberOfVisibleLinesAllowed, bool currentLineIsContentful);

    bool shouldDiscardRemainingContentInBlockDirection(size_t numberOfLinesWithInlineContent) const;

private:
    InlineLayoutUnit contentLeftAfterLastLine(const ConstraintsForInFlowContent&, std::optional<InlineLayoutUnit> lastLineLogicalBottom, const FloatingContext&) const;
    bool isAtSoftWrapOpportunity(const InlineItem& previous, const InlineItem& next) const;

    const InlineFormattingContext& formattingContext() const { return m_inlineFormattingContext; }

private:
    const InlineFormattingContext& m_inlineFormattingContext;
};

}
}

