/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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

#include "RenderTableSection.h"

namespace WebCore {

static const unsigned unsetRowIndex = 0x7FFFFFFF;
static const unsigned maxRowIndex = 0x7FFFFFFE; // 2,147,483,646

class RenderTableRow final : public RenderBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTableRow);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTableRow);
public:
    RenderTableRow(Element&, RenderStyle&&);
    RenderTableRow(Document&, RenderStyle&&);
    virtual ~RenderTableRow();

    RenderTableRow* nextRow() const;
    RenderTableRow* previousRow() const;

    RenderTableCell* firstCell() const;
    RenderTableCell* lastCell() const;

    RenderTable* table() const;

    void paintOutlineForRowIfNeeded(PaintInfo&, const LayoutPoint&);

    static RenderPtr<RenderTableRow> createAnonymousWithParentRenderer(const RenderTableSection&);
    RenderPtr<RenderBox> createAnonymousBoxWithSameTypeAs(const RenderBox&) const override;

    void setRowIndex(unsigned);
    bool rowIndexWasSet() const { return m_rowIndex != unsetRowIndex; }
    unsigned rowIndex() const;

    inline const BorderValue& borderAdjoiningTableStart() const;
    inline const BorderValue& borderAdjoiningTableEnd() const;
    const BorderValue& borderAdjoiningStartCell(const RenderTableCell&) const;
    const BorderValue& borderAdjoiningEndCell(const RenderTableCell&) const;

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    RenderTableSection* section() const { return downcast<RenderTableSection>(parent()); }

    void didInsertTableCell(RenderTableCell& child, RenderObject* beforeChild);

    // Whether a row has opaque background depends on many factors, e.g. border spacing, border collapsing, missing cells, etc.
    // For simplicity, just conservatively assume all table rows are not opaque.
    bool foregroundIsKnownToBeOpaqueInRect(const LayoutRect&, unsigned) const override { return false; }
    bool backgroundIsKnownToBeOpaqueInRect(const LayoutRect&) const override { return false; }

private:
    static RenderPtr<RenderTableRow> createTableRowWithStyle(Document&, const RenderStyle&);

    ASCIILiteral renderName() const override { return (isAnonymous() || isPseudoElement()) ? "RenderTableRow (anonymous)"_s : "RenderTableRow"_s; }
    bool canHaveChildren() const override { return true; }
    void willBeRemovedFromTree() override;
    void layout() override;

    LayoutRect clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext) const override;
    RepaintRects rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const override;
    void computeIntrinsicLogicalWidths(LayoutUnit&, LayoutUnit&) const override { }

    bool requiresLayer() const final;
    void paint(PaintInfo&, const LayoutPoint&) override;
    void imageChanged(WrappedImagePtr, const IntRect* = 0) override;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void firstChild() const = delete;
    void lastChild() const = delete;
    void nextSibling() const = delete;
    void previousSibling() const = delete;

    unsigned m_rowIndex : 31;
};

inline void RenderTableRow::setRowIndex(unsigned rowIndex)
{
    if (UNLIKELY(rowIndex > maxRowIndex))
        CRASH();
    m_rowIndex = rowIndex;
}

inline unsigned RenderTableRow::rowIndex() const
{
    ASSERT(rowIndexWasSet());
    return m_rowIndex;
}

inline RenderTable* RenderTableRow::table() const
{
    RenderTableSection* section = this->section();
    if (!section)
        return nullptr;
    return downcast<RenderTable>(section->parent());
}

inline RenderTableRow* RenderTableRow::nextRow() const
{
    return downcast<RenderTableRow>(RenderBox::nextSibling());
}

inline RenderTableRow* RenderTableRow::previousRow() const
{
    return downcast<RenderTableRow>(RenderBox::previousSibling());
}

inline RenderTableRow* RenderTableSection::firstRow() const
{
    return downcast<RenderTableRow>(RenderBox::firstChild());
}

inline RenderTableRow* RenderTableSection::lastRow() const
{
    return downcast<RenderTableRow>(RenderBox::lastChild());
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTableRow, isRenderTableRow())
