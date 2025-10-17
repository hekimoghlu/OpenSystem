/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#include "InlineContentCache.h"

namespace WebCore {
namespace Layout {

class InlineContentBreaker;
struct CandidateTextContent;
struct TextOnlyLineBreakResult;

class TextOnlySimpleLineBuilder final : public AbstractLineBuilder {
    WTF_MAKE_FAST_ALLOCATED;
public:
    TextOnlySimpleLineBuilder(InlineFormattingContext&, const ElementBox& rootBox, HorizontalConstraints rootHorizontalConstraints, const InlineItemList&);
    LineLayoutResult layoutInlineContent(const LineInput&, const std::optional<PreviousLine>&) final;

    static bool isEligibleForSimplifiedTextOnlyInlineLayoutByContent(const InlineContentCache::InlineItems&, const PlacedFloats&);
    static bool isEligibleForSimplifiedInlineLayoutByStyle(const RenderStyle&);

private:
    InlineItemPosition placeInlineTextContent(const RenderStyle&, const InlineItemRange&);
    InlineItemPosition placeNonWrappingInlineTextContent(const RenderStyle&, const InlineItemRange&);
    TextOnlyLineBreakResult handleOverflowingTextContent(const RenderStyle&, const InlineContentBreaker::ContinuousContent&, const InlineItemRange&);
    TextOnlyLineBreakResult commitCandidateContent(const RenderStyle&, const CandidateTextContent&, const InlineItemRange&);
    void initialize(const InlineItemRange&, const InlineRect& initialLogicalRect, const std::optional<PreviousLine>&);
    void handleLineEnding(const RenderStyle&, InlineItemPosition, size_t layoutRangeEndIndex);
    size_t revertToTrailingItem(const RenderStyle&, const InlineItemRange&, const InlineTextItem&);
    size_t revertToLastNonOverflowingItem(const RenderStyle&, const InlineItemRange&);
    InlineLayoutUnit availableWidth() const;
    bool isWrappingAllowed() const { return m_isWrappingAllowed; }

private:
    bool m_isWrappingAllowed { false };
    InlineLayoutUnit m_trimmedTrailingWhitespaceWidth { 0.f };
    std::optional<InlineLayoutUnit> m_overflowContentLogicalWidth { };
};

}
}
