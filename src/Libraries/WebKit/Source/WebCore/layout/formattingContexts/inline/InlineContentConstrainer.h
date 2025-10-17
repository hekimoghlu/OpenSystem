/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

#include "FormattingConstraints.h"
#include "InlineFormattingContext.h"
#include "InlineFormattingUtils.h"
#include "InlineItem.h"
#include "InlineLineBuilder.h"
#include "InlineTextItem.h"
#include <optional>

namespace WebCore {
namespace Layout {

class InlineContentConstrainer {
public:
    InlineContentConstrainer(InlineFormattingContext&, const InlineItemList&, const HorizontalConstraints&);
    std::optional<Vector<LayoutUnit>> computeParagraphLevelConstraints(TextWrapStyle);

private:
    void initialize();

    std::optional<Vector<LayoutUnit>> balanceRangeWithLineRequirement(InlineItemRange, InlineLayoutUnit idealLineWidth, size_t numberOfLines, bool isFirstChunk);
    std::optional<Vector<LayoutUnit>> balanceRangeWithNoLineRequirement(InlineItemRange, InlineLayoutUnit idealLineWidth, bool isFirstChunk);
    std::optional<Vector<LayoutUnit>> prettifyRange(InlineItemRange, InlineLayoutUnit idealLineWidth, bool isFirstChunk);

    InlineLayoutUnit inlineItemWidth(size_t inlineItemIndex, bool useFirstLineStyle) const;
    bool shouldTrimLeading(size_t inlineItemIndex, bool useFirstLineStyle, bool isFirstLineInChunk) const;
    bool shouldTrimTrailing(size_t inlineItemIndex, bool useFirstLineStyle) const;
    Vector<size_t> computeBreakOpportunities(InlineItemRange) const;
    Vector<LayoutUnit> computeLineWidthsFromBreaks(InlineItemRange, const Vector<size_t>& breaks, bool isFirstChunk) const;
    InlineLayoutUnit computeTextIndent(std::optional<bool> previousLineEndsWithLineBreak) const;

    InlineFormattingContext& m_inlineFormattingContext;
    const InlineItemList& m_inlineItemList;
    const HorizontalConstraints& m_horizontalConstraints;

    Vector<InlineItemRange> m_originalLineInlineItemRanges;
    Vector<float> m_originalLineWidths;
    Vector<bool> m_originalLineEndsWithForcedBreak;
    Vector<InlineLayoutUnit> m_inlineItemWidths;
    Vector<InlineLayoutUnit> m_firstLineStyleInlineItemWidths;
    size_t m_numberOfLinesInOriginalLayout { 0 };
    size_t m_numberOfInlineItems { 0 };
    double m_maximumLineWidth { 0 };
    bool m_cannotConstrainContent { false };
    bool m_hasSingleLineVisibleContent { false };

    struct SlidingWidth {
        SlidingWidth(const InlineContentConstrainer&, const InlineItemList&, size_t start, size_t end, bool useFirstLineStyle, bool isFirstLineInChunk);
        InlineLayoutUnit width();
        void advanceStart();
        void advanceStartTo(size_t newStart);
        void advanceEnd();
        void advanceEndTo(size_t newEnd);

    private:
        const InlineContentConstrainer& m_inlineContentConstrainer;
        const InlineItemList& m_inlineItemList;
        size_t m_start { 0 };
        size_t m_end { 0 };
        bool m_useFirstLineStyle { false };
        bool m_isFirstLineInChunk { false };
        InlineLayoutUnit m_totalWidth { 0 };
        InlineLayoutUnit m_leadingTrimmableWidth { 0 };
        InlineLayoutUnit m_trailingTrimmableWidth { 0 };
        std::optional<size_t> m_firstLeadingNonTrimmedItem;
    };
};

}
}
