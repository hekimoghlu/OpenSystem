/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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

#include "FormattingContext.h"
#include "TableFormattingGeometry.h"
#include "TableFormattingQuirks.h"
#include "TableFormattingState.h"
#include "TableGrid.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
namespace Layout {

// This class implements the layout logic for table formatting contexts.
// https://www.w3.org/TR/CSS22/tables.html
class TableFormattingContext final : public FormattingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TableFormattingContext);
public:
    TableFormattingContext(const ElementBox& formattingContextRoot, TableFormattingState&);
    void layoutInFlowContent(const ConstraintsForInFlowContent&) override;
    LayoutUnit usedContentHeight() const override;

    const TableFormattingGeometry& formattingGeometry() const { return m_tableFormattingGeometry; }
    const TableFormattingQuirks& formattingQuirks() const { return m_tableFormattingQuirks; }
    const TableFormattingState& formattingState() const { return m_tableFormattingState; }

private:
    class TableLayout {
    public:
        TableLayout(const TableFormattingContext&, const TableGrid&);

        using DistributedSpaces = Vector<LayoutUnit>;
        DistributedSpaces distributedHorizontalSpace(LayoutUnit availableHorizontalSpace);
        DistributedSpaces distributedVerticalSpace(std::optional<LayoutUnit> availableVerticalSpace);

    private:
        const TableFormattingContext& formattingContext() const { return m_formattingContext; }

        const TableFormattingContext& m_formattingContext;
        const TableGrid& m_grid;
    };

    TableFormattingContext::TableLayout tableLayout() const { return TableLayout(*this, formattingState().tableGrid()); }

    IntrinsicWidthConstraints computedIntrinsicWidthConstraints() override;
    void setUsedGeometryForCells(LayoutUnit availableHorizontalSpace, std::optional<LayoutUnit> availableVerticalSpace);
    void setUsedGeometryForRows(LayoutUnit availableHorizontalSpace);
    void setUsedGeometryForSections(const ConstraintsForInFlowContent&);

    IntrinsicWidthConstraints computedPreferredWidthForColumns();
    void computeAndDistributeExtraSpace(LayoutUnit availableHorizontalSpace, std::optional<LayoutUnit> availableVerticalSpace);

    TableFormattingState& formattingState() { return m_tableFormattingState; }

    TableFormattingState& m_tableFormattingState;
    const TableFormattingGeometry m_tableFormattingGeometry;
    const TableFormattingQuirks m_tableFormattingQuirks;
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONTEXT(TableFormattingContext, isTableFormattingContext())

