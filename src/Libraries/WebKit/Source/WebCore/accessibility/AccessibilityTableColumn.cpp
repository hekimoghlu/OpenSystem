/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "AccessibilityTableColumn.h"

#include "AccessibilityTable.h"

namespace WebCore {

AccessibilityTableColumn::AccessibilityTableColumn(AXID axID)
    : AccessibilityMockObject(axID)
{
}

AccessibilityTableColumn::~AccessibilityTableColumn() = default;

Ref<AccessibilityTableColumn> AccessibilityTableColumn::create(AXID axID)
{
    return adoptRef(*new AccessibilityTableColumn(axID));
}

void AccessibilityTableColumn::setParent(AccessibilityObject* parent)
{
    AccessibilityMockObject::setParent(parent);
    
    clearChildren();
}
    
LayoutRect AccessibilityTableColumn::elementRect() const
{
    // This used to be cached during the call to addChildren(), but calling elementRect()
    // can invalidate elements, so its better to ask for this on demand.
    LayoutRect columnRect;
    auto childrenCopy = const_cast<AccessibilityTableColumn*>(this)->unignoredChildren(/* updateChildrenIfNeeded */ false);
    for (const auto& cell : childrenCopy)
        columnRect.unite(cell->elementRect());

    return columnRect;
}

void AccessibilityTableColumn::setColumnIndex(unsigned columnIndex)
{
    if (m_columnIndex == columnIndex)
        return;
    m_columnIndex = columnIndex;

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (auto* cache = axObjectCache())
        cache->columnIndexChanged(*this);
#endif
}

bool AccessibilityTableColumn::computeIsIgnored() const
{
#if PLATFORM(IOS_FAMILY) || USE(ATSPI)
    return true;
#endif
    
    return !m_parent || m_parent->isIgnored();
}
    
void AccessibilityTableColumn::addChildren()
{
    ASSERT(!m_childrenInitialized); 
    m_childrenInitialized = true;

    RefPtr parentTable = dynamicDowncast<AccessibilityTable>(m_parent.get());
    if (!parentTable || !parentTable->isExposable())
        return;

    int numRows = parentTable->rowCount();
    for (int i = 0; i < numRows; ++i) {
        RefPtr cell = parentTable->cellForColumnAndRow(m_columnIndex, i);
        if (!cell)
            continue;

        // make sure the last one isn't the same as this one (rowspan cells)
        if (m_children.size() > 0 && m_children.last().ptr() == cell.get())
            continue;

        addChild(*cell);
    }
}
    
} // namespace WebCore
