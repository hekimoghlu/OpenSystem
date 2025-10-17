/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#include "RenderTableCol.h"

#include "BorderData.h"
#include "HTMLNames.h"
#include "HTMLTableColElement.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderChildIterator.h"
#include "RenderIterator.h"
#include "RenderStyleInlines.h"
#include "RenderTable.h"
#include "RenderTableCaption.h"
#include "RenderTableCell.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderTableCol);

RenderTableCol::RenderTableCol(Element& element, RenderStyle&& style)
    : RenderBox(Type::TableCol, element, WTFMove(style))
{
    // init RenderObject attributes
    setInline(true); // our object is not Inline
    updateFromElement();
    ASSERT(isRenderTableCol());
}

RenderTableCol::RenderTableCol(Document& document, RenderStyle&& style)
    : RenderBox(Type::TableCol, document, WTFMove(style))
{
    setInline(true);
    ASSERT(isRenderTableCol());
}

RenderTableCol::~RenderTableCol() = default;

void RenderTableCol::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderBox::styleDidChange(diff, oldStyle);
    RenderTable* table = this->table();
    if (!table)
        return;
    // If border was changed, notify table.
    if (!oldStyle)
        return;
    if (!oldStyle->borderIsEquivalentForPainting(style())) {
        table->invalidateCollapsedBorders();
        return;
    }
    if (oldStyle->width() != style().width()) {
        table->recalcSectionsIfNeeded();
        for (auto& section : childrenOfType<RenderTableSection>(*table)) {
            unsigned nEffCols = table->numEffCols();
            for (unsigned j = 0; j < nEffCols; j++) {
                unsigned rowCount = section.numRows();
                for (unsigned i = 0; i < rowCount; i++) {
                    RenderTableCell* cell = section.primaryCellAt(i, j);
                    if (!cell)
                        continue;
                    cell->setPreferredLogicalWidthsDirty(true);
                }
            }
        }
    }
}

void RenderTableCol::updateFromElement()
{
    ASSERT(element());
    unsigned oldSpan = m_span;
    if (element()->hasTagName(colTag) || element()->hasTagName(colgroupTag)) {
        HTMLTableColElement& tc = static_cast<HTMLTableColElement&>(*element());
        m_span = tc.span();
    } else
        m_span = 1;
    if (m_span != oldSpan && parent()) {
        if (hasInitializedStyle())
            setNeedsLayoutAndPrefWidthsRecalc();
        if (RenderTable* table = this->table())
            table->invalidateColumns();
    }
}

void RenderTableCol::insertedIntoTree()
{
    RenderBox::insertedIntoTree();
    table()->addColumn(this);
}

void RenderTableCol::willBeRemovedFromTree()
{
    RenderBox::willBeRemovedFromTree();
    if (auto* table = this->table()) {
        // We only need to invalidate the column cache when only individual columns are being removed (as opposed to when the entire table is being collapsed).
        table->invalidateColumns();
    }
}

bool RenderTableCol::isChildAllowed(const RenderObject& child, const RenderStyle& style) const
{
    // We cannot use isTableColumn here as style() may return 0.
    return style.display() == DisplayType::TableColumn && child.isRenderTableCol();
}

bool RenderTableCol::canHaveChildren() const
{
    // Cols cannot have children. This is actually necessary to fix a bug
    // with libraries.uc.edu, which makes a <p> be a table-column.
    return isTableColumnGroup();
}

LayoutRect RenderTableCol::clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext context) const
{
    // For now, just repaint the whole table.
    // FIXME: Find a better way to do this, e.g., need to repaint all the cells that we
    // might have propagated a background color or borders into.
    // FIXME: check for repaintContainer each time here?

    auto* parentTable = table();
    if (!parentTable)
        return { };

    return parentTable->clippedOverflowRect(repaintContainer, context);
}

auto RenderTableCol::rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const -> RepaintRects
{
    // Ignore RepaintOutlineBounds because it doesn't make sense to use the table's outline bounds to repaint a column.
    return { clippedOverflowRect(repaintContainer, visibleRectContextForRepaint()) };
}

void RenderTableCol::imageChanged(WrappedImagePtr, const IntRect*)
{
    // FIXME: Repaint only the rect the image paints in.
    if (!parent())
        return;
    repaint();
}

void RenderTableCol::clearPreferredLogicalWidthsDirtyBits()
{
    setPreferredLogicalWidthsDirty(false);

    for (auto& child : childrenOfType<RenderObject>(*this))
        child.setPreferredLogicalWidthsDirty(false);
}

RenderTable* RenderTableCol::table() const
{
    auto table = parent();
    if (table && !is<RenderTable>(*table))
        table = table->parent();
    return dynamicDowncast<RenderTable>(table);
}

RenderTableCol* RenderTableCol::enclosingColumnGroup() const
{
    auto* parentColumnGroup = dynamicDowncast<RenderTableCol>(*parent());
    if (!parentColumnGroup)
        return nullptr;

    ASSERT(parentColumnGroup->isTableColumnGroup());
    ASSERT(isTableColumn());
    return parentColumnGroup;
}

RenderTableCol* RenderTableCol::nextColumn() const
{
    // If |this| is a column-group, the next column is the colgroup's first child column.
    if (RenderObject* firstChild = this->firstChild())
        return downcast<RenderTableCol>(firstChild);

    // Otherwise it's the next column along.
    RenderObject* next = nextSibling();

    // Failing that, the child is the last column in a column-group, so the next column is the next column/column-group after its column-group.
    if (!next && is<RenderTableCol>(*parent()))
        next = parent()->nextSibling();

    for (; next; next = next->nextSibling()) {
        if (auto* column = dynamicDowncast<RenderTableCol>(*next))
            return column;
    }

    return nullptr;
}

const BorderValue& RenderTableCol::borderAdjoiningCellStartBorder() const
{
    return style().borderStart(table()->writingMode());
}

const BorderValue& RenderTableCol::borderAdjoiningCellEndBorder() const
{
    return style().borderEnd(table()->writingMode());
}

const BorderValue& RenderTableCol::borderAdjoiningCellBefore(const RenderTableCell& cell) const
{
    ASSERT_UNUSED(cell, table()->colElement(cell.col() + cell.colSpan()) == this);
    return style().borderStart(table()->writingMode());
}

const BorderValue& RenderTableCol::borderAdjoiningCellAfter(const RenderTableCell& cell) const
{
    ASSERT_UNUSED(cell, table()->colElement(cell.col() - 1) == this);
    return style().borderEnd(table()->writingMode());
}

LayoutUnit RenderTableCol::offsetLeft() const
{
    return table()->offsetLeftForColumn(*this);
}

LayoutUnit RenderTableCol::offsetTop() const
{
    return table()->offsetTopForColumn(*this);
}

LayoutUnit RenderTableCol::offsetWidth() const
{
    return table()->offsetWidthForColumn(*this);
}

LayoutUnit RenderTableCol::offsetHeight() const
{
    return table()->offsetHeightForColumn(*this);
}

}
