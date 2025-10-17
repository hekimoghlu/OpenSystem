/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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

#include "BlockFormattingContext.h"
#include "BlockFormattingQuirks.h"
#include "TableWrapperBlockFormattingQuirks.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

// This class implements the special block formatting context layout logic for the table wrapper.
// https://www.w3.org/TR/CSS22/tables.html#model
class TableWrapperBlockFormattingContext final : public BlockFormattingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TableWrapperBlockFormattingContext);
public:
    TableWrapperBlockFormattingContext(const ElementBox& formattingContextRoot, BlockFormattingState&);

    void layoutInFlowContent(const ConstraintsForInFlowContent&) final;

    void setHorizontalConstraintsIgnoringFloats(const HorizontalConstraints& horizontalConstraints) { m_horizontalConstraintsIgnoringFloats = horizontalConstraints; }

    const TableWrapperQuirks& formattingQuirks() const { return m_tableWrapperFormattingQuirks; }

private:
    void layoutTableBox(const ElementBox& tableBox, const ConstraintsForInFlowContent&);

    void computeBorderAndPaddingForTableBox(const ElementBox&, const HorizontalConstraints&);
    void computeWidthAndMarginForTableBox(const ElementBox&, const HorizontalConstraints&);
    void computeHeightAndMarginForTableBox(const ElementBox&, const ConstraintsForInFlowContent&);

    HorizontalConstraints m_horizontalConstraintsIgnoringFloats;
    const TableWrapperQuirks m_tableWrapperFormattingQuirks;
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONTEXT(TableWrapperBlockFormattingContext, isTableWrapperBlockFormattingContext())

