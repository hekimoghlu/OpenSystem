/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "TableFormattingState.h"

#include "RenderObject.h"
#include "TableFormattingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TableFormattingState);

static UniqueRef<TableGrid> ensureTableGrid(const ElementBox& tableBox)
{
    auto tableGrid = makeUniqueRef<TableGrid>();
    auto& tableStyle = tableBox.style();
    auto shouldApplyBorderSpacing = tableStyle.borderCollapse() == BorderCollapse::Separate;
    tableGrid->setHorizontalSpacing(LayoutUnit { shouldApplyBorderSpacing ? tableStyle.horizontalBorderSpacing() : 0 });
    tableGrid->setVerticalSpacing(LayoutUnit { shouldApplyBorderSpacing ? tableStyle.verticalBorderSpacing() : 0 });

    auto* firstChild = tableBox.firstChild();
    if (!firstChild) {
        // The rare case of empty table.
        return tableGrid;
    }

    const Box* tableCaption = nullptr;
    const Box* colgroup = nullptr;
    // Table caption is an optional element; if used, it is always the first child of a <table>.
    if (firstChild->isTableCaption())
        tableCaption = firstChild;
    // The <colgroup> must appear after any optional <caption> element but before any <thead>, <th>, <tbody>, <tfoot> and <tr> element.
    auto* colgroupCandidate = firstChild;
    if (tableCaption)
        colgroupCandidate = tableCaption->nextSibling();
    if (colgroupCandidate->isTableColumnGroup())
        colgroup = colgroupCandidate;

    if (colgroup) {
        auto& columns = tableGrid->columns();
        for (auto* column = downcast<ElementBox>(*colgroup).firstChild(); column; column = column->nextSibling()) {
            ASSERT(column->isTableColumn());
            auto columnSpanCount = column->columnSpan();
            ASSERT(columnSpanCount > 0);
            while (columnSpanCount--)
                columns.addColumn(downcast<ElementBox>(*column));
        }
    }

    auto* firstSection = colgroup ? colgroup->nextSibling() : tableCaption ? tableCaption->nextSibling() : firstChild;
    for (auto* section = firstSection; section; section = section->nextSibling()) {
        ASSERT(section->isTableHeader() || section->isTableBody() || section->isTableFooter());
        for (auto* row = downcast<ElementBox>(*section).firstChild(); row; row = row->nextSibling()) {
            ASSERT(row->isTableRow());
            for (auto* cell = downcast<ElementBox>(*row).firstChild(); cell; cell = cell->nextSibling()) {
                ASSERT(cell->isTableCell());
                tableGrid->appendCell(downcast<ElementBox>(*cell));
            }
        }
    }
    return tableGrid;
}


TableFormattingState::TableFormattingState(LayoutState& layoutState, const ElementBox& tableBox)
    : FormattingState(Type::Table, layoutState)
    , m_tableGrid(ensureTableGrid(tableBox))
{
}

TableFormattingState::~TableFormattingState()
{
}

}
}
