/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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

#include "InlineContentCache.h"
#include "LineLayoutResult.h"

namespace WebCore {
namespace Layout {

class AbstractLineBuilder;
class InlineFormattingContext;

class IntrinsicWidthHandler {
public:
    IntrinsicWidthHandler(InlineFormattingContext&, const InlineContentCache::InlineItems&);

    InlineLayoutUnit minimumContentSize();
    InlineLayoutUnit maximumContentSize();

    std::optional<LineLayoutResult>& maximumIntrinsicWidthLineContent() { return m_maximumIntrinsicWidthResultForSingleLine; }

private:
    enum class MayCacheLayoutResult : bool { No, Yes };
    InlineLayoutUnit computedIntrinsicWidthForConstraint(IntrinsicWidthMode, AbstractLineBuilder&, MayCacheLayoutResult = MayCacheLayoutResult::No);
    InlineLayoutUnit simplifiedMinimumWidth(const ElementBox& root) const;
    InlineLayoutUnit simplifiedMaximumWidth(MayCacheLayoutResult = MayCacheLayoutResult::No);

    InlineFormattingContext& formattingContext();
    const InlineFormattingContext& formattingContext() const;
    const InlineContentCache& formattingState() const;
    const ElementBox& formattingContextRoot() const;
    const ElementBox& lineBuilerRoot() const;
    const InlineItemList& inlineItemList() const { return m_inlineItems.content(); }

private:
    InlineFormattingContext& m_inlineFormattingContext;
    const InlineContentCache::InlineItems& m_inlineItems;
    InlineItemRange m_inlineItemRange;
    bool m_mayUseSimplifiedTextOnlyInlineLayoutInRange { false };

    std::optional<InlineLayoutUnit> m_maximumContentWidthBetweenLineBreaks { };
    std::optional<LineLayoutResult> m_maximumIntrinsicWidthResultForSingleLine { };
};

}
}

