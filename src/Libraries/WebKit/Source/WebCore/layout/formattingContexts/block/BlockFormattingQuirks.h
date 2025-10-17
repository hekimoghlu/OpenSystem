/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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

#include "FormattingQuirks.h"

namespace WebCore {
namespace Layout {

class BlockFormattingContext;

class BlockFormattingQuirks : public FormattingQuirks {
public:
    BlockFormattingQuirks(const BlockFormattingContext&);

    std::optional<LayoutUnit> stretchedInFlowHeightIfApplicable(const ElementBox&, ContentHeightAndMargin) const;
    virtual LayoutUnit heightValueOfNearestContainingBlockWithFixedHeight(const Box&) const;
    static bool shouldIgnoreCollapsedQuirkMargin(const ElementBox&);
    static bool shouldCollapseMarginBeforeWithParentMarginBefore(const ElementBox&);
    static bool shouldCollapseMarginAfterWithParentMarginAfter(const ElementBox&);
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_QUIRKS(BlockFormattingQuirks, isBlockFormattingQuirks())

