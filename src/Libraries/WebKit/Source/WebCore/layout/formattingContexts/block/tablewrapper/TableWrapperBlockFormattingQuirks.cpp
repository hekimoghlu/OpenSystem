/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "TableWrapperBlockFormattingQuirks.h"

#include "BlockFormattingGeometry.h"
#include "LayoutState.h"
#include "RenderObject.h"
#include "TableWrapperBlockFormattingContext.h"

namespace WebCore {
namespace Layout {

TableWrapperQuirks::TableWrapperQuirks(const TableWrapperBlockFormattingContext& formattingContext)
    : BlockFormattingQuirks(formattingContext)
{
}

LayoutUnit TableWrapperQuirks::overriddenTableHeight(const ElementBox& tableBox) const
{
    ASSERT(layoutState().inQuirksMode());
    // In quirks mode always use the content height. Note that the tables with content take computed values into account.
    auto& formattingContext = downcast<BlockFormattingContext>(this->formattingContext());
    return formattingContext.formattingGeometry().contentHeightForFormattingContextRoot(tableBox);
}

}
}

