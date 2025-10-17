/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#include "AccessibilityARIAGridRow.h"

#include "AccessibilityObject.h"
#include "AccessibilityTable.h"

namespace WebCore {
    
AccessibilityARIAGridRow::AccessibilityARIAGridRow(AXID axID, RenderObject& renderer)
    : AccessibilityTableRow(axID, renderer)
{
}

AccessibilityARIAGridRow::AccessibilityARIAGridRow(AXID axID, Node& node)
    : AccessibilityTableRow(axID, node)
{
}

AccessibilityARIAGridRow::~AccessibilityARIAGridRow() = default;

Ref<AccessibilityARIAGridRow> AccessibilityARIAGridRow::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityARIAGridRow(axID, renderer));
}

Ref<AccessibilityARIAGridRow> AccessibilityARIAGridRow::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityARIAGridRow(axID, node));
}

bool AccessibilityARIAGridRow::isARIATreeGridRow() const
{
    RefPtr parent = parentTable();
    if (!parent)
        return false;
    
    return parent->isTreeGrid();
}
    
AXCoreObject::AccessibilityChildrenVector AccessibilityARIAGridRow::disclosedRows()
{
    AccessibilityChildrenVector disclosedRows;
    // The contiguous disclosed rows will be the rows in the table that 
    // have an aria-level of plus 1 from this row.
    RefPtr parent = parentObjectUnignored();
    if (auto* axTable = dynamicDowncast<AccessibilityTable>(*parent); !axTable || !axTable->isExposable())
        return disclosedRows;

    // Search for rows that match the correct level. 
    // Only take the subsequent rows from this one that are +1 from this row's level.
    int rowIndex = this->rowIndex();
    if (rowIndex < 0)
        return disclosedRows;

    unsigned level = hierarchicalLevel();
    auto allRows = parent->rows();
    int rowCount = allRows.size();
    for (int k = rowIndex + 1; k < rowCount; ++k) {
        auto& row = allRows[k].get();
        // Stop at the first row that doesn't match the correct level.
        if (row.hierarchicalLevel() != level + 1)
            break;

        disclosedRows.append(row);
    }
    return disclosedRows;
}
    
AccessibilityObject* AccessibilityARIAGridRow::disclosedByRow() const
{
    // The row that discloses this one is the row in the table
    // that is aria-level subtract 1 from this row.
    RefPtr parent = dynamicDowncast<AccessibilityTable>(parentObjectUnignored());
    if (!parent || !parent->isExposable())
        return nullptr;

    // If the level is 1 or less, than nothing discloses this row.
    unsigned level = hierarchicalLevel();
    if (level <= 1)
        return nullptr;

    // Search for the previous row that matches the correct level.
    int index = rowIndex();
    auto allRows = parent->rows();
    int rowCount = allRows.size();
    if (index >= rowCount)
        return nullptr;

    for (int k = index - 1; k >= 0; --k) {
        auto& row = allRows[k].get();
        if (row.hierarchicalLevel() == level - 1)
            return &downcast<AccessibilityObject>(row);
    }
    return nullptr;
}

AccessibilityTable* AccessibilityARIAGridRow::parentTable() const
{
    // The parent table might not be the direct ancestor of the row unfortunately. ARIA states that role="grid" should
    // only have "row" elements, but if not, we still should handle it gracefully by finding the right table.
    return downcast<AccessibilityTable>(Accessibility::findAncestor<AccessibilityObject>(*this, false, [this] (const auto& ancestor) {
        // The parent table for an ARIA grid row should be an ARIA table.
        // Unless the row is a native tr element.
        if (auto* ancestorTable = dynamicDowncast<AccessibilityTable>(ancestor))
            return ancestorTable->isExposable() && (ancestorTable->isAriaTable() || node()->hasTagName(HTMLNames::trTag));

        return false;
    }));
}

AccessibilityObject* AccessibilityARIAGridRow::rowHeader()
{
    for (const auto& child : unignoredChildren()) {
        if (child->roleValue() == AccessibilityRole::RowHeader)
            return &downcast<AccessibilityObject>(child.get());
    }
    return nullptr;
}

} // namespace WebCore
