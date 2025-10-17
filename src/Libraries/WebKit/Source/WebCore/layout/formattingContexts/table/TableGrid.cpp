/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#include "TableGrid.h"

#include "RenderObject.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TableGrid);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TableGridCell);

TableGrid::Column::Column(const ElementBox* columnBox)
    : m_layoutBox(columnBox)
{
}

void TableGrid::Columns::addColumn(const ElementBox& columnBox)
{
    m_columnList.append({ &columnBox });
}

void TableGrid::Columns::addAnonymousColumn()
{
    m_columnList.append({ nullptr });
}

void TableGrid::Rows::addRow(const ElementBox& rowBox)
{
    m_rowList.append({ rowBox });
}

TableGrid::Row::Row(const ElementBox& rowBox)
    : m_layoutBox(rowBox)
{
}

TableGridCell::TableGridCell(const ElementBox& cellBox, SlotPosition position, CellSpan span)
    : m_layoutBox(cellBox)
    , m_position(position)
    , m_span(span)
{
}

TableGrid::Slot::Slot(TableGridCell& cell, bool isColumnSpanned, bool isRowSpanned)
    : m_cell(cell)
    , m_isColumnSpanned(isColumnSpanned)
    , m_isRowSpanned(isRowSpanned)
{
}

TableGrid::TableGrid()
{
}

TableGrid::Slot* TableGrid::slot(SlotPosition position)
{
    return m_slotMap.get(position);
}

void TableGrid::appendCell(const ElementBox& cellBox)
{
    auto rowSpan = cellBox.rowSpan();
    auto columnSpan = cellBox.columnSpan();
    auto isInNewRow = !cellBox.previousSibling();
    auto initialSlotPosition = SlotPosition { };

    if (!m_cells.isEmpty()) {
        auto& lastCell = m_cells.last();
        auto lastSlotPosition = lastCell->position();
        // First table cell in this row?
        if (isInNewRow)
            initialSlotPosition = SlotPosition { 0, lastSlotPosition.row + 1 };
        else
            initialSlotPosition = SlotPosition { lastSlotPosition.column + 1, lastSlotPosition.row };

        // Pick the next available slot by avoiding row and column spanners.
        while (true) {
            if (!m_slotMap.contains(initialSlotPosition))
                break;
            ++initialSlotPosition.column;
        }
    }
    auto cell = makeUnique<TableGridCell>(cellBox, initialSlotPosition, CellSpan { columnSpan, rowSpan });
    // Row and column spanners create additional slots.
    for (size_t row = 0; row < rowSpan; ++row) {
        for (auto column = cell->startColumn(); column < cell->endColumn(); ++column) {
            auto position = SlotPosition { column, cell->startRow() + row };
            ASSERT(!m_slotMap.contains(position));
            // This slot is spanned by a cell at the initial slow position.
            auto isColumnSpanned = column != cell->startColumn();
            auto isRowSpanned = !!row;
            m_slotMap.add(position, makeUnique<Slot>(*cell, isColumnSpanned, isRowSpanned));
        }
    }
    // Initialize columns/rows if needed.
    auto missingNumberOfColumns = std::max<int>(0, initialSlotPosition.column + columnSpan - m_columns.size());
    for (auto column = 0; column < missingNumberOfColumns; ++column)
        m_columns.addAnonymousColumn();

    if (isInNewRow)
        m_rows.addRow(cellBox.parent());

    m_cells.add(WTFMove(cell));
}

void TableGrid::insertCell(const ElementBox& cellBox, const ElementBox& before)
{
    UNUSED_PARAM(cellBox);
    UNUSED_PARAM(before);
}

void TableGrid::removeCell(const ElementBox& cellBox)
{
    UNUSED_PARAM(cellBox);
}

}
}
