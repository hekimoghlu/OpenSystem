/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#include "AbstractLineBuilder.h"
#include "FloatingContext.h"

namespace WebCore {
namespace Layout {

struct LineContent;
struct LineCandidate;

class LineBuilder final : public AbstractLineBuilder {
    WTF_MAKE_FAST_ALLOCATED;
public:
    LineBuilder(InlineFormattingContext&, HorizontalConstraints rootHorizontalConstraints, const InlineItemList&, TextSpacingContext = { });
    virtual ~LineBuilder() { };
    LineLayoutResult layoutInlineContent(const LineInput&, const std::optional<PreviousLine>&) final;

private:
    void candidateContentForLine(LineCandidate&, size_t inlineItemIndex, const InlineItemRange& needsLayoutRange, InlineLayoutUnit currentLogicalRight);
    InlineLayoutUnit leadingPunctuationWidthForLineCandiate(size_t firstInlineTextItemIndex, size_t candidateContentStartIndex) const;
    InlineLayoutUnit trailingPunctuationOrStopOrCommaWidthForLineCandiate(size_t lastInlineTextItemIndex, size_t layoutRangeEnd) const;

    struct Result {
        InlineContentBreaker::IsEndOfLine isEndOfLine { InlineContentBreaker::IsEndOfLine::No };
        struct CommittedContentCount {
            size_t value { 0 };
            bool isRevert { false };
        };
        CommittedContentCount committedCount { };
        size_t partialTrailingContentLength { 0 };
        std::optional<InlineLayoutUnit> overflowLogicalWidth { };
    };
    enum MayOverConstrainLine : uint8_t { No, Yes, OnlyWhenFirstFloatOnLine };
    bool tryPlacingFloatBox(const Box&, MayOverConstrainLine);
    Result handleInlineContent(const InlineItemRange& needsLayoutRange, const LineCandidate&);
    Result processLineBreakingResult(const LineCandidate&, const InlineItemRange& layoutRange, const InlineContentBreaker::Result&);
    struct RectAndFloatConstraints {
        InlineRect logicalRect;
        OptionSet<UsedFloat> constrainedSideSet { };
    };
    RectAndFloatConstraints floatAvoidingRect(const InlineRect& lineLogicalRect, InlineLayoutUnit lineMarginStart) const;
    RectAndFloatConstraints adjustedLineRectWithCandidateInlineContent(const LineCandidate&) const;
    size_t rebuildLineWithInlineContent(const InlineItemRange& needsLayoutRange, const InlineItem& lastInlineItemToAdd);
    size_t rebuildLineForTrailingSoftHyphen(const InlineItemRange& layoutRange);
    void commitPartialContent(const InlineContentBreaker::ContinuousContent::RunList&, const InlineContentBreaker::Result::PartialTrailingContent&);
    void initialize(const InlineRect& initialLineLogicalRect, const InlineItemRange& needsLayoutRange, const std::optional<PreviousLine>&,  std::optional<bool> previousLineEndsWithLineBreak);
    UniqueRef<LineContent> placeInlineAndFloatContent(const InlineItemRange&);
    struct InitialLetterOffsets {
        LayoutUnit capHeightOffset;
        LayoutUnit sunkenBelowFirstLineOffset;
    };
    std::optional<InitialLetterOffsets> adjustLineRectForInitialLetterIfApplicable(const Box& floatBox);
    bool isLastLineWithInlineContent(const LineContent&, size_t needsLayoutEnd, const Line::RunList&) const;

    bool isFloatLayoutSuspended() const { return !m_suspendedFloats.isEmpty(); }
    bool shouldTryToPlaceFloatBox(const Box& floatBox, LayoutUnit floatBoxMarginBoxWidth, MayOverConstrainLine) const;

    bool isLineConstrainedByFloat() const { return !m_lineIsConstrainedByFloat.isEmpty(); }
    const FloatingContext& floatingContext() const { return m_floatingContext; }

private:
    const FloatingContext& m_floatingContext;
    InlineRect m_lineInitialLogicalRect;
    InlineLayoutUnit m_lineMarginStart { 0.f };
    InlineLayoutUnit m_initialIntrusiveFloatsWidth { 0.f };
    InlineLayoutUnit m_candidateContentMaximumHeight { 0.f };
    LineLayoutResult::PlacedFloatList m_placedFloats;
    LineLayoutResult::SuspendedFloatList m_suspendedFloats;
    std::optional<InlineLayoutUnit> m_overflowingLogicalWidth;
    Vector<InlineItem, 1> m_lineSpanningInlineBoxes;
    OptionSet<UsedFloat> m_lineIsConstrainedByFloat { };
    std::optional<InlineLayoutUnit> m_initialLetterClearGap;
    TextSpacingContext m_textSpacingContext { };
};

}
}
