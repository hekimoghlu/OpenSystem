/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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

#include "RenderBlockFlow.h"
#include "RenderTableRow.h"
#include "RenderTableSection.h"

namespace WebCore {

// These is limited by the size of RenderTableCell::m_column bitfield.
static const unsigned unsetColumnIndex = 0x1FFFFFF;
static const unsigned maxColumnIndex = 0x1FFFFFE; // 33554430

enum IncludeBorderColorOrNot { DoNotIncludeBorderColor, IncludeBorderColor };

class RenderTableCell final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTableCell);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTableCell);
public:
    RenderTableCell(Element&, RenderStyle&&);
    RenderTableCell(Document&, RenderStyle&&);
    virtual ~RenderTableCell();
    
    unsigned colSpan() const;
    unsigned rowSpan() const;

    // Called from HTMLTableCellElement.
    void colSpanOrRowSpanChanged();

    void setCol(unsigned column);
    unsigned col() const;

    RenderTableCell* nextCell() const;
    RenderTableCell* previousCell() const;

    RenderTableRow* row() const { return downcast<RenderTableRow>(parent()); }
    RenderTableSection* section() const;
    RenderTable* table() const;
    unsigned rowIndex() const;
    inline Length styleOrColLogicalWidth() const;
    inline LayoutUnit logicalHeightForRowSizing() const;

    void setCellLogicalWidth(LayoutUnit constrainedLogicalWidth);

    RectEdges<LayoutUnit> borderWidths() const override;
    LayoutUnit borderLeft() const override;
    LayoutUnit borderRight() const override;
    LayoutUnit borderTop() const override;
    LayoutUnit borderBottom() const override;
    LayoutUnit borderStart() const override;
    LayoutUnit borderEnd() const override;
    LayoutUnit borderBefore() const override;
    LayoutUnit borderAfter() const override;

    void collectBorderValues(RenderTable::CollapsedBorderValues&) const;
    static void sortBorderValues(RenderTable::CollapsedBorderValues&);

    void layout() override;

    void paint(PaintInfo&, const LayoutPoint&) override;

    void paintCollapsedBorders(PaintInfo&, const LayoutPoint&);
    void paintBackgroundsBehindCell(PaintInfo&, LayoutPoint paintOffset, RenderBox* backgroundObject, LayoutPoint backgroundPaintOffset);

    LayoutUnit cellBaselinePosition() const;
    bool isBaselineAligned() const;

    void computeIntrinsicPadding(LayoutUnit rowHeight);
    void clearIntrinsicPadding() { setIntrinsicPadding(0, 0); }

    LayoutUnit intrinsicPaddingBefore() const { return m_intrinsicPaddingBefore; }
    LayoutUnit intrinsicPaddingAfter() const { return m_intrinsicPaddingAfter; }

    RectEdges<LayoutUnit> padding() const override;
    LayoutUnit paddingTop() const override;
    LayoutUnit paddingBottom() const override;
    LayoutUnit paddingLeft() const override;
    LayoutUnit paddingRight() const override;
    
    // FIXME: For now we just assume the cell has the same block flow direction as the table. It's likely we'll
    // create an extra anonymous RenderBlock to handle mixing directionality anyway, in which case we can lock
    // the block flow directionality of the cells to the table's directionality.
    LayoutUnit paddingBefore() const override;
    LayoutUnit paddingAfter() const override;

    void setOverridingLogicalHeightFromRowHeight(LayoutUnit);

    void scrollbarsChanged(bool horizontalScrollbarChanged, bool verticalScrollbarChanged) override;

    bool cellWidthChanged() const { return m_cellWidthChanged; }
    void setCellWidthChanged(bool b = true) { m_cellWidthChanged = b; }

    static RenderPtr<RenderTableCell> createAnonymousWithParentRenderer(const RenderTableRow&);
    RenderPtr<RenderBox> createAnonymousBoxWithSameTypeAs(const RenderBox&) const override;

    // Table layout always uses the table's writing mode.
    const WritingMode tableWritingMode() const { return table()->writingMode(); }

    inline const BorderValue& borderAdjoiningTableStart() const;
    inline const BorderValue& borderAdjoiningTableEnd() const;
    inline const BorderValue& borderAdjoiningCellBefore(const RenderTableCell&);
    inline const BorderValue& borderAdjoiningCellAfter(const RenderTableCell&);

    using RenderBlockFlow::nodeAtPoint;
#if ASSERT_ENABLED
    bool isFirstOrLastCellInRow() const { return !table()->cellAfter(this) || !table()->cellBefore(this); }
#endif
    
    RepaintRects localRectsForRepaint(RepaintOutlineBounds) const override;

    void invalidateHasEmptyCollapsedBorders();
    void setHasEmptyCollapsedBorder(CollapsedBorderSide, bool empty) const;

private:
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;
    void computePreferredLogicalWidths() override;

    LayoutRect frameRectForStickyPositioning() const override;

    static RenderPtr<RenderTableCell> createTableCellWithStyle(Document&, const RenderStyle&);

    ASCIILiteral renderName() const override { return (isAnonymous() || isPseudoElement()) ? "RenderTableCell (anonymous)"_s : "RenderTableCell"_s; }

    void willBeRemovedFromTree() override;

    void updateLogicalWidth() override;

    void paintBoxDecorations(PaintInfo&, const LayoutPoint&) override;
    void paintMask(PaintInfo&, const LayoutPoint&) override;

    LayoutSize offsetFromContainer(RenderElement&, const LayoutPoint&, bool* offsetDependsOnPoint = 0) const override;
    std::optional<RepaintRects> computeVisibleRectsInContainer(const RepaintRects&, const RenderLayerModelObject* container, VisibleRectContext) const override;

    LayoutUnit borderHalfLeft(bool outer) const;
    LayoutUnit borderHalfRight(bool outer) const;
    LayoutUnit borderHalfTop(bool outer) const;
    LayoutUnit borderHalfBottom(bool outer) const;

    LayoutUnit borderHalfStart(bool outer) const;
    LayoutUnit borderHalfEnd(bool outer) const;
    LayoutUnit borderHalfBefore(bool outer) const;
    LayoutUnit borderHalfAfter(bool outer) const;

    void setIntrinsicPaddingBefore(LayoutUnit p) { m_intrinsicPaddingBefore = p; }
    void setIntrinsicPaddingAfter(LayoutUnit p) { m_intrinsicPaddingAfter = p; }
    void setIntrinsicPadding(LayoutUnit before, LayoutUnit after) { setIntrinsicPaddingBefore(before); setIntrinsicPaddingAfter(after); }

    bool hasStartBorderAdjoiningTable() const;
    bool hasEndBorderAdjoiningTable() const;

    CollapsedBorderValue collapsedStartBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;
    CollapsedBorderValue collapsedEndBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;
    CollapsedBorderValue collapsedBeforeBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;
    CollapsedBorderValue collapsedAfterBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;

    CollapsedBorderValue cachedCollapsedLeftBorder(const WritingMode) const;
    CollapsedBorderValue cachedCollapsedRightBorder(const WritingMode) const;
    CollapsedBorderValue cachedCollapsedTopBorder(const WritingMode) const;
    CollapsedBorderValue cachedCollapsedBottomBorder(const WritingMode) const;

    CollapsedBorderValue computeCollapsedStartBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;
    CollapsedBorderValue computeCollapsedEndBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;
    CollapsedBorderValue computeCollapsedBeforeBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;
    CollapsedBorderValue computeCollapsedAfterBorder(IncludeBorderColorOrNot = IncludeBorderColor) const;

    Length logicalWidthFromColumns(RenderTableCol* firstColForThisCell, Length widthFromStyle) const;

    void updateColAndRowSpanFlags();

    unsigned parseRowSpanFromDOM() const;
    unsigned parseColSpanFromDOM() const;

    void nextSibling() const = delete;
    void previousSibling() const = delete;

    bool hasLineIfEmpty() const final;

    // Note MSVC will only pack members if they have identical types, hence we use unsigned instead of bool here.
    unsigned m_column : 25;
    unsigned m_cellWidthChanged : 1;
    unsigned m_hasColSpan: 1;
    unsigned m_hasRowSpan: 1;
    mutable unsigned m_hasEmptyCollapsedBeforeBorder: 1;
    mutable unsigned m_hasEmptyCollapsedAfterBorder: 1;
    mutable unsigned m_hasEmptyCollapsedStartBorder: 1;
    mutable unsigned m_hasEmptyCollapsedEndBorder: 1;
    LayoutUnit m_intrinsicPaddingBefore { 0 };
    LayoutUnit m_intrinsicPaddingAfter { 0 };
};

inline RenderTableCell* RenderTableCell::nextCell() const
{
    return downcast<RenderTableCell>(RenderBlockFlow::nextSibling());
}

inline RenderTableCell* RenderTableCell::previousCell() const
{
    return downcast<RenderTableCell>(RenderBlockFlow::previousSibling());
}

inline unsigned RenderTableCell::colSpan() const
{
    if (!m_hasColSpan)
        return 1;
    return parseColSpanFromDOM();
}

inline unsigned RenderTableCell::rowSpan() const
{
    if (!m_hasRowSpan)
        return 1;
    return parseRowSpanFromDOM();
}

inline void RenderTableCell::setCol(unsigned column)
{
    if (UNLIKELY(column > maxColumnIndex))
        column = maxColumnIndex;
    m_column = column;
}

inline unsigned RenderTableCell::col() const
{
    ASSERT(m_column != unsetColumnIndex);
    return m_column;
}

inline RenderTableSection* RenderTableCell::section() const
{
    RenderTableRow* row = this->row();
    if (!row)
        return nullptr;
    return downcast<RenderTableSection>(row->parent());
}

inline RenderTable* RenderTableCell::table() const
{
    RenderTableSection* section = this->section();
    if (!section)
        return nullptr;
    return downcast<RenderTable>(section->parent());
}

inline unsigned RenderTableCell::rowIndex() const
{
    // This function shouldn't be called on a detached cell.
    ASSERT(row());
    return row()->rowIndex();
}

inline RenderTableCell* RenderTableRow::firstCell() const
{
    return downcast<RenderTableCell>(RenderBox::firstChild());
}

inline RenderTableCell* RenderTableRow::lastCell() const
{
    return downcast<RenderTableCell>(RenderBox::lastChild());
}

inline void RenderTableCell::setHasEmptyCollapsedBorder(CollapsedBorderSide side, bool empty) const
{
    switch (side) {
    case CBSAfter: {
        m_hasEmptyCollapsedAfterBorder = empty;
        break;
    }
    case CBSBefore: {
        m_hasEmptyCollapsedBeforeBorder = empty;
        break;
    }
    case CBSStart: {
        m_hasEmptyCollapsedStartBorder = empty;
        break;
    }
    case CBSEnd: {
        m_hasEmptyCollapsedEndBorder = empty;
        break;
    }
    }
    if (empty)
        table()->collapsedEmptyBorderIsPresent();
}

inline void RenderTableCell::invalidateHasEmptyCollapsedBorders()
{
    m_hasEmptyCollapsedBeforeBorder = false;
    m_hasEmptyCollapsedAfterBorder = false;
    m_hasEmptyCollapsedStartBorder = false;
    m_hasEmptyCollapsedEndBorder = false;
}

inline RenderPtr<RenderBox> RenderTableCell::createAnonymousBoxWithSameTypeAs(const RenderBox& renderer) const
{
    return RenderTableCell::createTableCellWithStyle(renderer.document(), renderer.style());
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTableCell, isRenderTableCell())
