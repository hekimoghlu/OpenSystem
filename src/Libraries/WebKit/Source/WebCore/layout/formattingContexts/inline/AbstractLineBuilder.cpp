/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#include "AbstractLineBuilder.h"

#include "FontCascade.h"
#include "InlineContentBreaker.h"
#include "InlineFormattingContext.h"
#include "RenderStyleInlines.h"

namespace WebCore {
namespace Layout {

AbstractLineBuilder::AbstractLineBuilder(InlineFormattingContext& inlineFormattingContext, const ElementBox& rootBox, HorizontalConstraints rootHorizontalConstraints, const InlineItemList& inlineItemList)
    : m_line(inlineFormattingContext)
    , m_inlineItemList(inlineItemList.span())
    , m_inlineFormattingContext(inlineFormattingContext)
    , m_rootBox(rootBox)
    , m_rootHorizontalConstraints(rootHorizontalConstraints)
{
}

void AbstractLineBuilder::reset()
{
    m_wrapOpportunityList.shrink(0);
    m_partialLeadingTextItem = { };
    m_previousLine = { };
}

std::optional<InlineLayoutUnit> AbstractLineBuilder::eligibleOverflowWidthAsLeading(const InlineContentBreaker::ContinuousContent::RunList& candidateRuns, const InlineContentBreaker::Result& lineBreakingResult, bool isFirstFormattedLine) const
{
    auto eligibleTrailingRunIndex = [&]() -> std::optional<size_t> {
        ASSERT(lineBreakingResult.action == InlineContentBreaker::Result::Action::Wrap || lineBreakingResult.action == InlineContentBreaker::Result::Action::Break);
        if (candidateRuns.size() == 1 && candidateRuns.first().inlineItem.isText()) {
            // A single text run is always a candidate.
            return { 0 };
        }
        if (lineBreakingResult.action == InlineContentBreaker::Result::Action::Break && lineBreakingResult.partialTrailingContent) {
            auto& trailingRun = candidateRuns[lineBreakingResult.partialTrailingContent->trailingRunIndex];
            if (trailingRun.inlineItem.isText())
                return lineBreakingResult.partialTrailingContent->trailingRunIndex;
        }
        return { };
    }();

    if (!eligibleTrailingRunIndex)
        return { };

    auto& overflowingRun = candidateRuns[*eligibleTrailingRunIndex];
    // FIXME: Add support for other types of continuous content.
    ASSERT(is<InlineTextItem>(overflowingRun.inlineItem));
    auto& inlineTextItem = downcast<InlineTextItem>(overflowingRun.inlineItem);
    if (inlineTextItem.isWhitespace())
        return { };
    if (isFirstFormattedLine) {
        auto& usedStyle = overflowingRun.style;
        auto& style = overflowingRun.inlineItem.style();
        if (&usedStyle != &style && !usedStyle.fontCascadeEqual(style)) {
            // We may have the incorrect text width when styles differ. Just re-measure the text content when we place it on the next line.
            return { };
        }
    }
    auto logicalWidthForNextLineAsLeading = overflowingRun.contentWidth();
    if (lineBreakingResult.action == InlineContentBreaker::Result::Action::Wrap)
        return logicalWidthForNextLineAsLeading;
    if (lineBreakingResult.action == InlineContentBreaker::Result::Action::Break && lineBreakingResult.partialTrailingContent->partialRun)
        return logicalWidthForNextLineAsLeading - lineBreakingResult.partialTrailingContent->partialRun->logicalWidth;
    return { };
}

void AbstractLineBuilder::setIntrinsicWidthMode(IntrinsicWidthMode intrinsicWidthMode)
{
    m_intrinsicWidthMode = intrinsicWidthMode;
    m_inlineContentBreaker.setIsMinimumInIntrinsicWidthMode(m_intrinsicWidthMode == IntrinsicWidthMode::Minimum);
}

const RenderStyle& AbstractLineBuilder::rootStyle() const
{
    return isFirstFormattedLine() ? root().firstLineStyle() : root().style();
}

const InlineLayoutState& AbstractLineBuilder::layoutState() const
{
    return formattingContext().layoutState();
}

InlineLayoutState& AbstractLineBuilder::layoutState()
{
    return formattingContext().layoutState();
}

}
}
