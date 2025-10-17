/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

#include "LayoutUnit.h"
#include "Length.h"
#include "TableLayout.h"
#include <wtf/Vector.h>

namespace WebCore {

class RenderTable;
class RenderTableCell;

class AutoTableLayout final : public TableLayout {
public:
    explicit AutoTableLayout(RenderTable*);
    virtual ~AutoTableLayout();

    void computeIntrinsicLogicalWidths(LayoutUnit& minWidth, LayoutUnit& maxWidth, TableIntrinsics) override;
    LayoutUnit scaledWidthFromPercentColumns() const override { return m_scaledWidthFromPercentColumns; }
    void applyPreferredLogicalWidthQuirks(LayoutUnit& minWidth, LayoutUnit& maxWidth) const override;
    void layout() override;

private:
    void fullRecalc();
    void recalcColumn(unsigned effCol);

    float calcEffectiveLogicalWidth();

    void insertSpanCell(RenderTableCell*);

    struct Layout {
        Length logicalWidth;
        Length effectiveLogicalWidth;
        float minLogicalWidth { 0 };
        float maxLogicalWidth { 0 };
        float effectiveMinLogicalWidth { 0 };
        float effectiveMaxLogicalWidth { 0 };
        float computedLogicalWidth { 0 };
        bool emptyCellsOnly { true };
    };

    Vector<Layout> m_layoutStruct;
    Vector<RenderTableCell*> m_spanCells;
    bool m_hasPercent : 1;
    mutable bool m_effectiveLogicalWidthDirty : 1;
    LayoutUnit m_scaledWidthFromPercentColumns;
};

} // namespace WebCore
