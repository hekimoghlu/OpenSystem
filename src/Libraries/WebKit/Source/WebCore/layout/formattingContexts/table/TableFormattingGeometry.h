/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

#include "FormattingGeometry.h"
#include "TableGrid.h"

namespace WebCore {
namespace Layout {

class TableFormattingContext;

class TableFormattingGeometry : public FormattingGeometry {
public:
    TableFormattingGeometry(const TableFormattingContext&);

    LayoutUnit cellBoxContentHeight(const ElementBox&) const;
    BoxGeometry::Edges computedCellBorder(const TableGridCell&) const;
    std::optional<LayoutUnit> computedColumnWidth(const ElementBox& columnBox) const;
    IntrinsicWidthConstraints intrinsicWidthConstraintsForCellContent(const TableGridCell&) const;
    InlineLayoutUnit usedBaselineForCell(const ElementBox& cellBox) const;
    LayoutUnit horizontalSpaceForCellContent(const TableGridCell&) const;
    LayoutUnit verticalSpaceForCellContent(const TableGridCell&, std::optional<LayoutUnit> availableVerticalSpace) const;

private:
    const TableFormattingContext& formattingContext() const { return downcast<TableFormattingContext>(FormattingGeometry::formattingContext()); }
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_GEOMETRY(TableFormattingGeometry, isTableFormattingGeometry())

