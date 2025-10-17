/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#include "AccessibilityRenderObject.h"

namespace WebCore {
    
class AccessibilityTable;
class AccessibilityTableRow;

class AccessibilityTableCell : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityTableCell> create(AXID, RenderObject&);
    static Ref<AccessibilityTableCell> create(AXID, Node&);
    virtual ~AccessibilityTableCell();
    bool isTableCell() const final { return true; }

    bool isExposedTableCell() const final;
    bool isTableHeaderCell() const;
    bool isColumnHeader() const final;
    bool isRowHeader() const final;

    std::optional<AXID> rowGroupAncestorID() const final;

    virtual AccessibilityTable* parentTable() const;

    // Returns the start location and row span of the cell.
    std::pair<unsigned, unsigned> rowIndexRange() const final;
    // Returns the start location and column span of the cell.
    std::pair<unsigned, unsigned> columnIndexRange() const final;

    AccessibilityChildrenVector rowHeaders() final;

    int axColumnIndex() const final;
    int axRowIndex() const final;
    unsigned colSpan() const;
    unsigned rowSpan() const;
    void incrementEffectiveRowSpan() { ++m_effectiveRowSpan; }
    void resetEffectiveRowSpan() { m_effectiveRowSpan = 1; }
    void setAXColIndexFromRow(int index) { m_axColIndexFromRow = index; }

    void setRowIndex(unsigned);
    void setColumnIndex(unsigned);

#if USE(ATSPI)
    int axColumnSpan() const;
    int axRowSpan() const;
#endif

protected:
    explicit AccessibilityTableCell(AXID, RenderObject&);
    explicit AccessibilityTableCell(AXID, Node&);

    AccessibilityTableRow* parentRow() const;
    AccessibilityRole determineAccessibilityRole() final;

private:
    // If a table cell is not exposed as a table cell, a TH element can serve as its title UI element.
    AccessibilityObject* titleUIElement() const final;
    bool computeIsIgnored() const final;
    String expandedTextValue() const final;
    bool supportsExpandedTextValue() const final;
    void ensureIndexesUpToDate() const;

    unsigned m_rowIndex { 0 };
    unsigned m_columnIndex { 0 };
    int m_axColIndexFromRow { -1 };

    // How many rows does this cell actually span?
    // This differs from rowSpan(), which can be an author-specified number all the way up 65535 that doesn't actually
    // reflect how many rows the cell spans in the rendered table.
    // Default to 1, as the cell should span at least the row it starts in.
    unsigned m_effectiveRowSpan { 1 };
};

} // namespace WebCore 

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityTableCell, isTableCell())
