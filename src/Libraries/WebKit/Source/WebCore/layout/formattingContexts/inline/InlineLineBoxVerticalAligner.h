/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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

#include "InlineFormattingContext.h"
#include "InlineFormattingUtils.h"

namespace WebCore {
namespace Layout {

class LineBoxVerticalAligner {
public:
    LineBoxVerticalAligner(const InlineFormattingContext&);
    InlineLayoutUnit computeLogicalHeightAndAlign(LineBox&) const;

private:
    InlineLayoutUnit simplifiedVerticalAlignment(LineBox&) const;

    struct LineBoxAlignmentContent {
        InlineLayoutUnit height() const { return std::max(nonLineBoxRelativeAlignedMaximumHeight, std::max(topAndBottomAlignedMaximumHeight.top.value_or(0.f), topAndBottomAlignedMaximumHeight.bottom.value_or(0.f))); }

        InlineLayoutUnit nonLineBoxRelativeAlignedMaximumHeight { 0 };
        struct TopAndBottomAlignedMaximumHeight {
            std::optional<InlineLayoutUnit> top { };
            std::optional<InlineLayoutUnit> bottom { };
        };
        TopAndBottomAlignedMaximumHeight topAndBottomAlignedMaximumHeight { };
        bool hasTextEmphasis { false };
    };
    LineBoxAlignmentContent computeLineBoxLogicalHeight(LineBox&) const;
    void computeRootInlineBoxVerticalPosition(LineBox&, const LineBoxAlignmentContent&) const;
    void alignInlineLevelBoxes(LineBox&, InlineLayoutUnit lineBoxLogicalHeight) const;
    InlineLayoutUnit adjustForAnnotationIfNeeded(LineBox&, InlineLayoutUnit lineBoxHeight) const;
    std::optional<InlineLevelBox::AscentAndDescent> layoutBoundsForInlineBoxSubtree(const LineBox::InlineLevelBoxList& nonRootInlineLevelBoxes, size_t inlineBoxIndex) const;
    enum class IsInlineLeveBoxAlignment : bool { No, Yes };
    InlineLayoutUnit logicalTopOffsetFromParentBaseline(const InlineLevelBox&, const InlineLevelBox& parentInlineBox, IsInlineLeveBoxAlignment = IsInlineLeveBoxAlignment::No) const;

    const InlineFormattingUtils& formattingUtils() const { return formattingContext().formattingUtils(); }
    const InlineFormattingContext& formattingContext() const { return m_inlineFormattingContext; }
    const ElementBox& rootBox() const { return formattingContext().root(); }
    const InlineLayoutState& layoutState() const { return formattingContext().layoutState(); }

private:
    const InlineFormattingContext& m_inlineFormattingContext;
};

}
}

