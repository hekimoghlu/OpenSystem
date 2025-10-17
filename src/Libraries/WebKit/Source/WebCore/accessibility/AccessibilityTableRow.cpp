/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include "AccessibilityTableRow.h"

#include "AXObjectCache.h"
#include "AccessibilityTable.h"
#include "AccessibilityTableCell.h"
#include "HTMLNames.h"
#include "RenderObject.h"

namespace WebCore {

using namespace HTMLNames;

AccessibilityTableRow::AccessibilityTableRow(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityTableRow::AccessibilityTableRow(AXID axID, Node& node)
    : AccessibilityRenderObject(axID, node)
{
}

AccessibilityTableRow::~AccessibilityTableRow() = default;

Ref<AccessibilityTableRow> AccessibilityTableRow::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityTableRow(axID, renderer));
}

Ref<AccessibilityTableRow> AccessibilityTableRow::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityTableRow(axID, node));
}

AccessibilityRole AccessibilityTableRow::determineAccessibilityRole()
{
    if (!isTableRow())
        return AccessibilityRenderObject::determineAccessibilityRole();

    if ((m_ariaRole = determineAriaRoleAttribute()) != AccessibilityRole::Unknown)
        return m_ariaRole;

    return AccessibilityRole::Row;
}

bool AccessibilityTableRow::isTableRow() const
{
    auto* table = parentTable();
    return table && table->isExposable();
}
    
AccessibilityObject* AccessibilityTableRow::observableObject() const
{
    // This allows the table to be the one who sends notifications about tables.
    return parentTable();
}
    
bool AccessibilityTableRow::computeIsIgnored() const
{    
    AccessibilityObjectInclusion decision = defaultObjectInclusion();
    if (decision == AccessibilityObjectInclusion::IncludeObject)
        return false;
    if (decision == AccessibilityObjectInclusion::IgnoreObject)
        return true;
    
    if (!isTableRow())
        return AccessibilityRenderObject::computeIsIgnored();

    return isRenderHidden() || ignoredFromPresentationalRole();
}
    
AccessibilityTable* AccessibilityTableRow::parentTable() const
{
    // The parent table might not be the direct ancestor of the row unfortunately. ARIA states that role="grid" should
    // only have "row" elements, but if not, we still should handle it gracefully by finding the right table.
    for (RefPtr parent = parentObject(); parent; parent = parent->parentObject()) {
        // If this is a non-anonymous table object, but not an accessibility table, we should stop because we don't want to
        // choose another ancestor table as this row's table.
        if (auto* parentTable = dynamicDowncast<AccessibilityTable>(*parent)) {
            if (parentTable->isExposable())
                return parentTable;
            if (parentTable->node())
                break;
        }
    }
    return nullptr;
}

void AccessibilityTableRow::setRowIndex(unsigned rowIndex)
{
    if (m_rowIndex == rowIndex)
        return;
    m_rowIndex = rowIndex;

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (auto* cache = axObjectCache())
        cache->rowIndexChanged(*this);
#endif
}

AccessibilityObject* AccessibilityTableRow::rowHeader()
{
    const auto& rowChildren = unignoredChildren();
    if (rowChildren.isEmpty())
        return nullptr;
    
    Ref firstCell = rowChildren[0].get();
    if (!firstCell->node() || !firstCell->node()->hasTagName(thTag))
        return nullptr;

    // Verify that the row header is not part of an entire row of headers.
    // In that case, it is unlikely this is a row header.
    for (const auto& child : rowChildren) {
        // We found a non-header cell, so this is not an entire row of headers -- return the original header cell.
        if (child->node() && !child->node()->hasTagName(thTag))
            return &downcast<AccessibilityObject>(firstCell.get());
    }
    return nullptr;
}

void AccessibilityTableRow::addChildren()
{
    // If the element specifies its cells through aria-owns, return that first.
    auto ownedObjects = this->ownedObjects();
    if (ownedObjects.size()) {
        for (auto& object : ownedObjects)
            addChild(downcast<AccessibilityObject>(object.get()), DescendIfIgnored::No);
        m_childrenInitialized = true;
        m_subtreeDirty = false;
    }
    else
        AccessibilityRenderObject::addChildren();

    // "ARIA 1.1, If the set of columns which is present in the DOM is contiguous, and if there are no cells which span more than one row or
    // column in that set, then authors may place aria-colindex on each row, setting the value to the index of the first column of the set."
    // Update child cells' axColIndex if there's an aria-colindex value set for the row. So the cell doesn't have to go through the siblings
    // to calculate the index.
    int colIndex = axColumnIndex();
    if (colIndex == -1)
        return;

    unsigned index = 0;
    for (const auto& cell : unignoredChildren()) {
        if (auto* tableCell = dynamicDowncast<AccessibilityTableCell>(cell.get()))
            tableCell->setAXColIndexFromRow(colIndex + index);
        index++;
    }
}

int AccessibilityTableRow::axColumnIndex() const
{
    int value = getIntegralAttribute(aria_colindexAttr);
    return value >= 1 ? value : -1;
}

int AccessibilityTableRow::axRowIndex() const
{
    int value = getIntegralAttribute(aria_rowindexAttr);
    return value >= 1 ? value : -1;
}
    
} // namespace WebCore
