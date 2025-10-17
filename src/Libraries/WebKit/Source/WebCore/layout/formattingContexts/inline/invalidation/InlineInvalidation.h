/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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

#include "InlineDamage.h"
#include "InlineDisplayContent.h"
#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class RenderStyle;

namespace Layout {

class Box;
class ElementBox;
class InlineTextBox;
struct InvalidatedLine;

class InlineInvalidation {
public:
    InlineInvalidation(InlineDamage&, const InlineItemList&, const InlineDisplay::Content&);

    bool rootStyleWillChange(const ElementBox& formattingContextRoot, const RenderStyle& newStyle);
    bool styleWillChange(const Box&, const RenderStyle& newStyle, StyleDifference);

    bool textInserted(const InlineTextBox& newOrDamagedInlineTextBox, std::optional<size_t> offset = { });
    bool textWillBeRemoved(const InlineTextBox&, std::optional<size_t> offset = { });

    bool inlineLevelBoxInserted(const Box&);
    bool inlineLevelBoxWillBeRemoved(const Box&);
    bool inlineLevelBoxContentWillChange(const Box&);

    bool restartForPagination(size_t lineIndex, LayoutUnit pageTopAdjustment);

    static bool mayOnlyNeedPartialLayout(const InlineDamage* inlineDamage) { return inlineDamage && inlineDamage->layoutStartPosition(); }
    static void resetInlineDamage(InlineDamage&);

private:
    enum class ShouldApplyRangeLayout : bool { No, Yes };
    bool updateInlineDamage(const InvalidatedLine&, InlineDamage::Reason, ShouldApplyRangeLayout = ShouldApplyRangeLayout::No, LayoutUnit restartPaginationAdjustment = 0_lu);
    bool setFullLayoutIfNeeded(const Box&);
    const InlineDisplay::Boxes& displayBoxes() const { return m_displayContent.boxes; }
    const InlineDisplay::Lines& displayLines() const { return m_displayContent.lines; }

    InlineDamage& m_inlineDamage;

    const InlineItemList& m_inlineItemList;
    const InlineDisplay::Content& m_displayContent;
};

}
}
