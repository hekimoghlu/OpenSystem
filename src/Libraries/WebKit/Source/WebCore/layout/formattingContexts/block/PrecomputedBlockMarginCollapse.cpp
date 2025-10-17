/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "BlockMarginCollapse.h"

#include "BlockFormattingContext.h"
#include "BlockFormattingGeometry.h"
#include "BlockFormattingQuirks.h"
#include "LayoutBox.h"
#include "LayoutElementBox.h"
#include "LayoutState.h"
#include "LayoutUnit.h"
#include "RenderStyleInlines.h"

namespace WebCore {
namespace Layout {

UsedVerticalMargin::PositiveAndNegativePair::Values BlockMarginCollapse::precomputedPositiveNegativeValues(const ElementBox& layoutBox, const BlockFormattingGeometry& formattingGeometry) const
{
    if (formattingState().hasUsedVerticalMargin(layoutBox))
        return formattingState().usedVerticalMargin(layoutBox).positiveAndNegativeValues.before;

    auto horizontalConstraints = formattingGeometry.constraintsForInFlowContent(FormattingContext::containingBlock(layoutBox)).horizontal();
    auto computedVerticalMargin = formattingGeometry.computedVerticalMargin(layoutBox, horizontalConstraints);
    auto nonCollapsedMargin = UsedVerticalMargin::NonCollapsedValues { computedVerticalMargin.before.value_or(0), computedVerticalMargin.after.value_or(0) };
    return precomputedPositiveNegativeMarginBefore(layoutBox, nonCollapsedMargin, formattingGeometry);
}

UsedVerticalMargin::PositiveAndNegativePair::Values BlockMarginCollapse::precomputedPositiveNegativeMarginBefore(const ElementBox& layoutBox, UsedVerticalMargin::NonCollapsedValues nonCollapsedValues, const BlockFormattingGeometry& formattingGeometry) const
{
    auto firstChildCollapsedMarginBefore = [&]() -> UsedVerticalMargin::PositiveAndNegativePair::Values {
        if (!marginBeforeCollapsesWithFirstInFlowChildMarginBefore(layoutBox))
            return { };
        return precomputedPositiveNegativeValues(downcast<ElementBox>(*layoutBox.firstInFlowChild()), formattingGeometry);
    };

    auto previouSiblingCollapsedMarginAfter = [&]() -> UsedVerticalMargin::PositiveAndNegativePair::Values {
        if (!marginBeforeCollapsesWithPreviousSiblingMarginAfter(layoutBox))
            return { };
        auto& previousInFlowSibling = *layoutBox.previousInFlowSibling();
        return formattingState().usedVerticalMargin(previousInFlowSibling).positiveAndNegativeValues.after;
    };

    // 1. Gather positive and negative margin values from first child if margins are adjoining.
    // 2. Gather positive and negative margin values from previous inflow sibling if margins are adjoining.
    // 3. Compute min/max positive and negative collapsed margin values using non-collpased computed margin before.
    auto collapsedMarginBefore = computedPositiveAndNegativeMargin(firstChildCollapsedMarginBefore(), previouSiblingCollapsedMarginAfter());
    auto shouldIgnoreCollapsedMargin = collapsedMarginBefore.isQuirk && inQuirksMode() && BlockFormattingQuirks::shouldIgnoreCollapsedQuirkMargin(layoutBox);
    if (shouldIgnoreCollapsedMargin)
        collapsedMarginBefore = { };

    auto nonCollapsedBefore = UsedVerticalMargin::PositiveAndNegativePair::Values { };
    if (nonCollapsedValues.before > 0)
        nonCollapsedBefore = { nonCollapsedValues.before, { }, layoutBox.style().marginBefore().hasQuirk() };
    else if (nonCollapsedValues.before < 0)
        nonCollapsedBefore = { { }, nonCollapsedValues.before, layoutBox.style().marginBefore().hasQuirk() };

    return computedPositiveAndNegativeMargin(collapsedMarginBefore, nonCollapsedBefore);
}

PrecomputedMarginBefore BlockMarginCollapse::precomputedMarginBefore(const ElementBox& layoutBox, UsedVerticalMargin::NonCollapsedValues usedNonCollapsedMargin, const BlockFormattingGeometry& formattingGeometry)
{
    ASSERT(layoutBox.isBlockLevelBox());
    // Don't pre-compute vertical margins for out of flow boxes.
    ASSERT(layoutBox.isInFlow() || layoutBox.isFloatingPositioned());
    ASSERT(!layoutBox.isReplacedBox());

    auto positiveNegativeMarginBefore = precomputedPositiveNegativeMarginBefore(layoutBox, usedNonCollapsedMargin, formattingGeometry);
    auto collapsedMarginBefore = marginValue(positiveNegativeMarginBefore);
    return { usedNonCollapsedMargin.before, collapsedMarginBefore, positiveNegativeMarginBefore };
}

}
}
