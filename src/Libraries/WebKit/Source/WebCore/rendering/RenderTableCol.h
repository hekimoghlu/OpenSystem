/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

#include "RenderBox.h"

namespace WebCore {

class RenderTable;
class RenderTableCell;

class RenderTableCol final : public RenderBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTableCol);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTableCol);
public:
    RenderTableCol(Element&, RenderStyle&&);
    RenderTableCol(Document&, RenderStyle&&);
    virtual ~RenderTableCol();

    void clearPreferredLogicalWidthsDirtyBits();

    unsigned span() const { return m_span; }
    void setSpan(unsigned span) { m_span = span; }

    bool isTableColumnGroupWithColumnChildren() const { return firstChild(); }
    bool isTableColumn() const { return style().display() == DisplayType::TableColumn; }
    bool isTableColumnGroup() const { return style().display() == DisplayType::TableColumnGroup; }

    RenderTableCol* enclosingColumnGroup() const;
    RenderTableCol* enclosingColumnGroupIfAdjacentBefore() const;
    RenderTableCol* enclosingColumnGroupIfAdjacentAfter() const;

    // Returns the next column or column-group.
    RenderTableCol* nextColumn() const;

    const BorderValue& borderAdjoiningCellStartBorder() const;
    const BorderValue& borderAdjoiningCellEndBorder() const;
    const BorderValue& borderAdjoiningCellBefore(const RenderTableCell&) const;
    const BorderValue& borderAdjoiningCellAfter(const RenderTableCell&) const;

    LayoutUnit offsetLeft() const override;
    LayoutUnit offsetTop() const override;
    LayoutUnit offsetWidth() const override;
    LayoutUnit offsetHeight() const override;
    void updateFromElement() override;

private:
    ASCIILiteral renderName() const override { return "RenderTableCol"_s; }
    void computePreferredLogicalWidths() override { ASSERT_NOT_REACHED(); }
    void computeIntrinsicLogicalWidths(LayoutUnit&, LayoutUnit&) const override { ASSERT_NOT_REACHED(); }

    void insertedIntoTree() override;
    void willBeRemovedFromTree() override;

    bool isChildAllowed(const RenderObject&, const RenderStyle&) const override;
    bool canHaveChildren() const override;
    bool requiresLayer() const override { return false; }

    LayoutRect clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext) const override;
    RepaintRects rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const override;

    void imageChanged(WrappedImagePtr, const IntRect* = 0) override;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;
    void paint(PaintInfo&, const LayoutPoint&) override { }

    RenderTable* table() const;

    unsigned m_span { 1 };
};

inline RenderTableCol* RenderTableCol::enclosingColumnGroupIfAdjacentBefore() const
{
    if (previousSibling())
        return nullptr;
    return enclosingColumnGroup();
}

inline RenderTableCol* RenderTableCol::enclosingColumnGroupIfAdjacentAfter() const
{
    if (nextSibling())
        return nullptr;
    return enclosingColumnGroup();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTableCol, isRenderTableCol())
