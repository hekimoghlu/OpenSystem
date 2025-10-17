/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "TableFormattingQuirks.h"

#include "LayoutBox.h"
#include "LayoutContainingBlockChainIterator.h"
#include "LayoutElementBox.h"
#include "LayoutState.h"
#include "RenderStyleInlines.h"
#include "TableFormattingContext.h"
#include "TableGrid.h"

namespace WebCore {
namespace Layout {

TableFormattingQuirks::TableFormattingQuirks(const TableFormattingContext& tableFormattingContext)
    : FormattingQuirks(tableFormattingContext)
{
}

bool TableFormattingQuirks::shouldIgnoreChildContentVerticalMargin(const ElementBox& cellBox)
{
    // Normally BFC root content height takes the margin box of the child content as vertical margins don't collapse with BFC roots,
    // but table cell boxes do collapse their (non-existing) margins with child quirk margins (so much quirk), so here we check
    // if the content height should include margins or not.
    // e.g <table><tr><td><p>text content</td></tr></table> <- <p>'s quirk margin collapses with the <td> so its content
    // height should not include vertical margins.
    if (cellBox.establishesInlineFormattingContext())
        return false;
    if (!cellBox.hasInFlowChild())
        return false;
    return cellBox.firstInFlowChild()->style().marginBefore().hasQuirk() || cellBox.lastInFlowChild()->style().marginAfter().hasQuirk();
}

LayoutUnit TableFormattingQuirks::heightValueOfNearestContainingBlockWithFixedHeight(const Box& layoutBox) const
{
    // The "let's find the nearest ancestor with fixed height to resolve percent height" quirk is limited to the table formatting
    // context. If we can't resolve it within the table subtree, we default it to 0.
    // e.g <div style="height: 100px"><table><tr><td style="height: 100%"></td></tr></table></div> is resolved to 0px.
    for (auto& ancestor : containingBlockChainWithinFormattingContext(layoutBox, formattingContext().root())) {
        auto height = ancestor.style().logicalHeight();
        if (height.isFixed())
            return LayoutUnit { height.value() };
    }
    return { };
}

}
}

