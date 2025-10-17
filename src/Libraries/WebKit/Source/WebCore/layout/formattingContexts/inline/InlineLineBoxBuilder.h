/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#include "InlineLayoutState.h"
#include "InlineLineBuilder.h"
#include "TextUtil.h"

namespace WebCore {
namespace Layout {

class Box;
class ElementBox;
class LayoutState;

class LineBoxBuilder {
public:
    LineBoxBuilder(const InlineFormattingContext&, LineLayoutResult&);

    LineBox build(size_t lineIndex);

private:
    void setVerticalPropertiesForInlineLevelBox(const LineBox&, InlineLevelBox&) const;
    void setLayoutBoundsForInlineBox(InlineLevelBox&, FontBaseline) const;
    void adjustInlineBoxHeightsForLineBoxContainIfApplicable(LineBox&);
    void computeLineBoxGeometry(LineBox&) const;
    InlineLevelBox::AscentAndDescent enclosingAscentDescentWithFallbackFonts(const InlineLevelBox&, const TextUtil::FallbackFontList& fallbackFontsForContent, FontBaseline) const;
    TextUtil::FallbackFontList collectFallbackFonts(const InlineLevelBox& parentInlineBox, const Line::Run&, const RenderStyle&);
    void adjustMarginStartForListMarker(const ElementBox& listMarkerBox, LayoutUnit nestedListMarkerMarginStart, InlineLayoutUnit rootInlineBoxOffset) const;
    InlineLayoutUnit applyTextBoxTrimOnLineBoxIfNeeded(InlineLayoutUnit lineBoxLogicalHeight, InlineLevelBox& rootInlineBox) const;

    void constructInlineLevelBoxes(LineBox&);
    void adjustIdeographicBaselineIfApplicable(LineBox&);
    void adjustOutsideListMarkersPosition(LineBox&);
    void expandAboveRootInlineBox(LineBox&, InlineLayoutUnit) const;

    bool isFirstLine() const { return lineLayoutResult().isFirstLast.isFirstFormattedLine != LineLayoutResult::IsFirstLast::FirstFormattedLine::No; }
    bool isLastLine() const { return lineLayoutResult().isFirstLast.isLastLineWithInlineContent; }
    const InlineFormattingContext& formattingContext() const { return m_inlineFormattingContext; }
    const LineLayoutResult& lineLayoutResult() const { return m_lineLayoutResult; }
    const ElementBox& rootBox() const { return formattingContext().root(); }
    const RenderStyle& rootStyle() const { return isFirstLine() ? rootBox().firstLineStyle() : rootBox().style(); }

    const InlineLayoutState& layoutState() const { return formattingContext().layoutState(); }
    const BlockLayoutState& blockLayoutState() const { return layoutState().parentBlockLayoutState(); }

private:
    const InlineFormattingContext& m_inlineFormattingContext;
    LineLayoutResult& m_lineLayoutResult;
    bool m_fallbackFontRequiresIdeographicBaseline { false };
    bool m_lineHasNonLineSpanningRubyContent { false };
    UncheckedKeyHashMap<const InlineLevelBox*, TextUtil::FallbackFontList> m_fallbackFontsForInlineBoxes;
    Vector<size_t> m_outsideListMarkers;
};

}
}

