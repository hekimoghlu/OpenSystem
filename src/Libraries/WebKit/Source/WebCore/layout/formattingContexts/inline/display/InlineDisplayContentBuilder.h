/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include "InlineLineBuilder.h"
#include "LayoutUnits.h"
#include <wtf/Range.h>

namespace WebCore {
namespace Layout {

struct AncestorStack;
class ElementBox;
struct DisplayBoxTree;
struct IsFirstLastIndex;
class LineBox;

class InlineDisplayContentBuilder {
public:
    InlineDisplayContentBuilder(InlineFormattingContext&, const ConstraintsForInlineContent&, const LineBox&, const InlineDisplay::Line&);

    InlineDisplay::Boxes build(const LineLayoutResult&);

private:
    void processNonBidiContent(const LineLayoutResult&, InlineDisplay::Boxes&);
    void processBidiContent(const LineLayoutResult&, InlineDisplay::Boxes&);
    void collectInkOverflowForInlineBoxes(InlineDisplay::Boxes&);
    void collectInkOverflowForTextDecorations(InlineDisplay::Boxes&);
    void truncateForEllipsisPolicy(LineEndingTruncationPolicy, const LineLayoutResult&, InlineDisplay::Boxes&);

    void appendTextDisplayBox(const Line::Run&, const InlineRect&, InlineDisplay::Boxes&);
    void appendSoftLineBreakDisplayBox(const Line::Run&, const InlineRect&, InlineDisplay::Boxes&);
    void appendHardLineBreakDisplayBox(const Line::Run&, const InlineRect&, InlineDisplay::Boxes&);
    void appendAtomicInlineLevelDisplayBox(const Line::Run&, const InlineRect&, InlineDisplay::Boxes&);
    void appendRootInlineBoxDisplayBox(const InlineRect&, bool lineHasContent, InlineDisplay::Boxes&);
    void appendInlineBoxDisplayBox(const Line::Run&, const InlineLevelBox&, const InlineRect&, InlineDisplay::Boxes&);
    void appendInlineDisplayBoxAtBidiBoundary(const Box&, InlineDisplay::Boxes&);
    void insertRubyAnnotationBox(const Box& annotationBox, size_t insertionPosition, const InlineRect&, InlineDisplay::Boxes&);

    size_t processRubyBase(size_t rubyBaseStart, InlineDisplay::Boxes&, Vector<WTF::Range<size_t>>& interlinearRubyColumnRangeList, Vector<size_t>& rubyBaseStartIndexListWithAnnotation);
    void processRubyContent(InlineDisplay::Boxes&, const LineLayoutResult&);

    inline InlineRect mapInlineRectLogicalToVisual(const InlineRect& logicalRect, const InlineRect& containerLogicalRect, WritingMode);

    void setInlineBoxGeometry(const Box& inlineBox, Layout::BoxGeometry&, const InlineRect&, bool isFirstInlineBoxFragment);
    void adjustVisualGeometryForDisplayBox(size_t displayBoxNodeIndex, InlineLayoutUnit& accumulatedOffset, InlineLayoutUnit lineBoxLogicalTop, const DisplayBoxTree&, InlineDisplay::Boxes&, const UncheckedKeyHashMap<const Box*, IsFirstLastIndex>&);
    size_t ensureDisplayBoxForContainer(const ElementBox&, DisplayBoxTree&, AncestorStack&, InlineDisplay::Boxes&);

    InlineRect flipLogicalRectToVisualForWritingModeWithinLine(const InlineRect& logicalRect, const InlineRect& lineLogicalRect, WritingMode) const;
    InlineRect flipRootInlineBoxRectToVisualForWritingMode(const InlineRect& rootInlineBoxLogicalRect, WritingMode) const;
    template <typename BoxType, typename LayoutUnitType>
    void setLogicalLeft(BoxType&, LayoutUnitType logicalLeft, WritingMode) const;
    void setLogicalRight(InlineDisplay::Box&, InlineLayoutUnit logicalRight, WritingMode) const;
    InlineLayoutPoint movePointHorizontallyForWritingMode(const InlineLayoutPoint& topLeft, InlineLayoutUnit horizontalOffset, WritingMode) const;
    InlineLayoutUnit outsideListMarkerVisualPosition(const ElementBox&) const;
    void setGeometryForBlockLevelOutOfFlowBoxes(const Vector<size_t>& indexList, const Line::RunList&, const Vector<int32_t>& visualOrderList = { });

    bool isLineFullyTruncatedInBlockDirection() const { return m_lineIsFullyTruncatedInBlockDirection; }

    const LineBox& lineBox() const { return m_lineBox; }
    size_t lineIndex() const { return lineBox().lineIndex(); }
    const ConstraintsForInlineContent& constraints() const { return m_constraints; }
    const ElementBox& root() const { return m_formattingContext.root(); }
    const RenderStyle& rootStyle() const { return lineIndex() ? root().style() : root().firstLineStyle(); }
    InlineFormattingContext& formattingContext() { return m_formattingContext; }
    const InlineFormattingContext& formattingContext() const { return m_formattingContext; }

private:
    InlineFormattingContext& m_formattingContext;
    const ConstraintsForInlineContent& m_constraints;
    const LineBox& m_lineBox;
    const InlineDisplay::Line& m_displayLine;
    IntSize m_initialContaingBlockSize;
    // FIXME: This should take DisplayLine::isFullyTruncatedInBlockDirection() for non-prefixed line-clamp.
    bool m_lineIsFullyTruncatedInBlockDirection { false };
    bool m_contentHasInkOverflow { false };
    bool m_hasSeenRubyBase { false };
    bool m_hasSeenTextDecoration { false };
};

inline InlineRect InlineDisplayContentBuilder::mapInlineRectLogicalToVisual(const InlineRect& logicalRect, const InlineRect& containerLogicalRect, WritingMode writingMode)
{
    InlineRect visualRect = logicalRect;
    switch (writingMode.computedWritingMode()) {
    case StyleWritingMode::HorizontalTb:
        return visualRect;

    case StyleWritingMode::HorizontalBt:
        visualRect.setLeft(logicalRect.left());
        visualRect.setTop(containerLogicalRect.height() - logicalRect.bottom());
        return visualRect;

    case StyleWritingMode::VerticalRl:
    case StyleWritingMode::SidewaysRl:
        visualRect.setLeft(logicalRect.top());
        visualRect.setTop(logicalRect.left());
        visualRect.setWidth(logicalRect.height());
        visualRect.setHeight(logicalRect.width());
        return visualRect;

    case StyleWritingMode::VerticalLr:
        visualRect.setLeft(containerLogicalRect.height() - logicalRect.bottom());
        visualRect.setTop(logicalRect.left());
        visualRect.setWidth(logicalRect.height());
        visualRect.setHeight(logicalRect.width());
        return visualRect;

    case StyleWritingMode::SidewaysLr:
        visualRect.setLeft(logicalRect.top());
        visualRect.setTop(containerLogicalRect.width() - logicalRect.right());
        visualRect.setWidth(logicalRect.height());
        visualRect.setHeight(logicalRect.width());
        return visualRect;

    default:
        ASSERT_NOT_REACHED();
        return visualRect;
    }
}

}
}

