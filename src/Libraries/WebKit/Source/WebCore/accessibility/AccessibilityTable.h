/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include <wtf/Forward.h>

namespace WebCore {

class AccessibilityTableRow;
class HTMLTableElement;

class AccessibilityTable : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityTable> create(AXID, RenderObject&);
    static Ref<AccessibilityTable> create(AXID, Node&);
    virtual ~AccessibilityTable();

    void init() final;

    virtual bool isAriaTable() const { return false; }

    void addChildren() final;
    void clearChildren() final;
    void updateChildrenRoles();

    AccessibilityChildrenVector columns() final;
    AccessibilityChildrenVector rows() final;

    unsigned columnCount() final;
    unsigned rowCount() final;

    String title() const final;

    // all the cells in the table
    AccessibilityChildrenVector cells() final;
    AccessibilityObject* cellForColumnAndRow(unsigned column, unsigned row) final;

    AccessibilityChildrenVector rowHeaders() final;
    AccessibilityChildrenVector visibleRows() final;

    // Returns an object that contains, as children, all the objects that act as headers.
    AccessibilityObject* headerContainer() final;

    bool isTable() const final { return true; }
    // Returns whether it is exposed as an AccessibilityTable to the platform.
    bool isExposable() const final { return m_isExposable; }
    void recomputeIsExposable();

    int axColumnCount() const final;
    int axRowCount() const final;

    // Cell indexes are assigned during child creation, so make sure children are up-to-date.
    void ensureCellIndexesUpToDate() { updateChildrenIfNecessary(); }
    Vector<Vector<Markable<AXID>>> cellSlots() final;
    void setCellSlotsDirty();

protected:
    explicit AccessibilityTable(AXID, RenderObject&);
    explicit AccessibilityTable(AXID, Node&);

    AccessibilityChildrenVector m_rows;
    AccessibilityChildrenVector m_columns;
    // 2D matrix of the cells assigned to each "slot" in this table.
    // ("Slot" as defined here: https://html.spec.whatwg.org/multipage/tables.html#concept-slots)
    Vector<Vector<Markable<AXID>>> m_cellSlots;

    RefPtr<AccessibilityObject> m_headerContainer;
    bool m_isExposable;

    // Used in type checking function is<AccessibilityTable>.
    bool isAccessibilityTableInstance() const final { return true; }

    bool computeIsIgnored() const final;

    void addRow(AccessibilityTableRow&, unsigned, unsigned& maxColumnCount);

private:
    AccessibilityRole determineAccessibilityRole() final;
    virtual bool computeIsTableExposableThroughAccessibility() const { return isDataTable(); }
    void labelText(Vector<AccessibilityText>&) const final;
    HTMLTableElement* tableElement() const;

    // Returns the number of columns the table should have.
    unsigned computeCellSlots();

    void ensureRow(unsigned);
    void ensureRowAndColumn(unsigned /* rowIndex */, unsigned /* columnIndex */);

    bool hasNonTableARIARole() const;
    // isDataTable is whether it is exposed as an AccessibilityTable because the heuristic
    // think this "looks" like a data-based table (instead of a table used for layout).
    bool isDataTable() const;
};

} // namespace WebCore 

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityTable, isAccessibilityTableInstance())
