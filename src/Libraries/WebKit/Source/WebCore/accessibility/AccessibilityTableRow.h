/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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

class AccessibilityTableRow : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityTableRow> create(AXID, RenderObject&);
    static Ref<AccessibilityTableRow> create(AXID, Node&);
    virtual ~AccessibilityTableRow();

    // retrieves the "row" header (a th tag in the rightmost column)
    AccessibilityObject* rowHeader() override;
    virtual AccessibilityTable* parentTable() const;

    void setRowIndex(unsigned);
    unsigned rowIndex() const override { return m_rowIndex; }

    // allows the table to add other children that may not originate
    // in the row, but their col/row spans overlap into it
    void appendChild(AccessibilityObject*);
    
    void addChildren() final;

    int axColumnIndex() const final;
    int axRowIndex() const final;

protected:
    explicit AccessibilityTableRow(AXID, RenderObject&);
    explicit AccessibilityTableRow(AXID, Node&);

    AccessibilityRole determineAccessibilityRole() final;

private:
    // FIXME: This implementation of isTableRow() causes us to do an ancestry traversal every time is<AccessibilityTableRow>
    // is called. Can we replace this with a simpler check? And this function should then maybe be called isExposedTableRow()?
    bool isTableRow() const final;
    AccessibilityObject* observableObject() const final;
    bool computeIsIgnored() const final;

    unsigned m_rowIndex;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityTableRow, isTableRow())
