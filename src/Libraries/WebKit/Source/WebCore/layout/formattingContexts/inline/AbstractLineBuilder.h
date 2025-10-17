/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

#include "FormattingConstraints.h"
#include "InlineContentBreaker.h"
#include "InlineLayoutState.h"
#include "InlineLine.h"
#include "InlineLineTypes.h"
#include "LineLayoutResult.h"

namespace WebCore {
namespace Layout {

struct LineInput {
    InlineItemRange needsLayoutRange;
    InlineRect initialLogicalRect;
};

class AbstractLineBuilder {
public:
    virtual LineLayoutResult layoutInlineContent(const LineInput&, const std::optional<PreviousLine>&) = 0;
    virtual ~AbstractLineBuilder() { };

    void setIntrinsicWidthMode(IntrinsicWidthMode);

protected:
    AbstractLineBuilder(InlineFormattingContext&, const ElementBox& rootBox, HorizontalConstraints rootHorizontalConstraints, const InlineItemList&);

    void reset();

    std::optional<InlineLayoutUnit> eligibleOverflowWidthAsLeading(const InlineContentBreaker::ContinuousContent::RunList&, const InlineContentBreaker::Result&, bool) const;

    std::optional<IntrinsicWidthMode> intrinsicWidthMode() const { return m_intrinsicWidthMode; }
    bool isInIntrinsicWidthMode() const { return !!intrinsicWidthMode(); }

    bool isFirstFormattedLine() const { return !m_previousLine.has_value(); }

    InlineContentBreaker& inlineContentBreaker() { return m_inlineContentBreaker; }

    InlineFormattingContext& formattingContext() { return m_inlineFormattingContext; }
    const InlineFormattingContext& formattingContext() const { return m_inlineFormattingContext; }
    const HorizontalConstraints& rootHorizontalConstraints() const { return m_rootHorizontalConstraints; }
    const InlineLayoutState& layoutState() const;
    InlineLayoutState& layoutState();
    const BlockLayoutState& blockLayoutState() const { return layoutState().parentBlockLayoutState(); }
    const ElementBox& root() const { return m_rootBox; }
    const RenderStyle& rootStyle() const;

protected:
    Line m_line;
    InlineRect m_lineLogicalRect;
    std::span<const InlineItem> m_inlineItemList;
    Vector<const InlineItem*, 32> m_wrapOpportunityList;
    std::optional<InlineTextItem> m_partialLeadingTextItem;
    std::optional<PreviousLine> m_previousLine { };

private:
    InlineFormattingContext& m_inlineFormattingContext;
    const ElementBox& m_rootBox; // Note that this is not necessarily a block container (see range builder).
    HorizontalConstraints m_rootHorizontalConstraints;

    InlineContentBreaker m_inlineContentBreaker;
    std::optional<IntrinsicWidthMode> m_intrinsicWidthMode;
};


}
}
