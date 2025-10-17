/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#include "RenderMathMLPadded.h"

#if ENABLE(MATHML)

#include "RenderMathMLBlockInlines.h"
#include <cmath>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLPadded);

RenderMathMLPadded::RenderMathMLPadded(MathMLPaddedElement& element, RenderStyle&& style)
    : RenderMathMLRow(Type::MathMLPadded, element, WTFMove(style))
{
    ASSERT(isRenderMathMLPadded());
}

RenderMathMLPadded::~RenderMathMLPadded() = default;

LayoutUnit RenderMathMLPadded::voffset() const
{
    return toUserUnits(element().voffset(), style(), 0);
}

LayoutUnit RenderMathMLPadded::lspace() const
{
    LayoutUnit lspace = toUserUnits(element().lspace(), style(), 0);
    // FIXME: Negative lspace values are not supported yet (https://bugs.webkit.org/show_bug.cgi?id=85730).
    return std::max<LayoutUnit>(0, lspace);
}

LayoutUnit RenderMathMLPadded::mpaddedWidth(LayoutUnit contentWidth) const
{
    return std::max<LayoutUnit>(0, toUserUnits(element().width(), style(), contentWidth));
}

LayoutUnit RenderMathMLPadded::mpaddedHeight(LayoutUnit contentHeight) const
{
    return std::max<LayoutUnit>(0, toUserUnits(element().height(), style(), contentHeight));
}

LayoutUnit RenderMathMLPadded::mpaddedDepth(LayoutUnit contentDepth) const
{
    return std::max<LayoutUnit>(0, toUserUnits(element().depth(), style(), contentDepth));
}

void RenderMathMLPadded::computePreferredLogicalWidths()
{
    ASSERT(preferredLogicalWidthsDirty());

    // Only the width attribute should modify the width.
    // We parse it using the preferred width of the content as its default value.
    LayoutUnit preferredWidth = preferredLogicalWidthOfRowItems();
    preferredWidth = mpaddedWidth(preferredWidth);
    m_maxPreferredLogicalWidth = m_minPreferredLogicalWidth = preferredWidth;

    auto sizes = sizeAppliedToMathContent(LayoutPhase::CalculatePreferredLogicalWidth);
    applySizeToMathContent(LayoutPhase::CalculatePreferredLogicalWidth, sizes);

    adjustPreferredLogicalWidthsForBorderAndPadding();

    setPreferredLogicalWidthsDirty(false);
}

void RenderMathMLPadded::layoutBlock(bool relayoutChildren, LayoutUnit)
{
    ASSERT(needsLayout());

    insertPositionedChildrenIntoContainingBlock();

    if (!relayoutChildren && simplifiedLayout())
        return;

    layoutFloatingChildren();

    recomputeLogicalWidth();
    computeAndSetBlockDirectionMarginsOfChildren();

    // We first layout our children as a normal <mrow> element.
    LayoutUnit contentWidth, contentAscent, contentDescent;
    stretchVerticalOperatorsAndLayoutChildren();
    getContentBoundingBox(contentWidth, contentAscent, contentDescent);
    layoutRowItems(contentWidth, contentAscent);

    // We parse the mpadded attributes using the content metrics as the default value.
    LayoutUnit width = mpaddedWidth(contentWidth);
    LayoutUnit ascent = mpaddedHeight(contentAscent);
    LayoutUnit descent = mpaddedDepth(contentDescent);

    // Align children on the new baseline and shift them by (lspace, -voffset)
    shiftInFlowChildren(lspace(), ascent - contentAscent - voffset());

    // Set the final metrics.
    setLogicalWidth(width);
    setLogicalHeight(ascent + descent);

    auto sizes = sizeAppliedToMathContent(LayoutPhase::Layout);
    auto shift = applySizeToMathContent(LayoutPhase::Layout, sizes);
    shiftInFlowChildren(shift, 0);

    adjustLayoutForBorderAndPadding();

    layoutPositionedObjects(relayoutChildren);

    updateScrollInfoAfterLayout();

    clearNeedsLayout();
}

std::optional<LayoutUnit> RenderMathMLPadded::firstLineBaseline() const
{
    // We try and calculate the baseline from the position of the first child.
    LayoutUnit ascent;
    if (auto* baselineChild = firstInFlowChildBox())
        ascent = ascentForChild(*baselineChild) + baselineChild->logicalTop() + voffset();
    else
        ascent = mpaddedHeight(0);
    return ascent;
}

}

#endif
