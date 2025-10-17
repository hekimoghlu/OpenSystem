/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#include "RenderMathMLRow.h"

#if ENABLE(MATHML)

#include "MathMLNames.h"
#include "MathMLRowElement.h"
#include "RenderIterator.h"
#include "RenderMathMLBlockInlines.h"
#include "RenderMathMLOperator.h"
#include "RenderMathMLRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace MathMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLRow);

RenderMathMLRow::RenderMathMLRow(Type type, MathMLRowElement& element, RenderStyle&& style)
    : RenderMathMLBlock(type, element, WTFMove(style))
{
    ASSERT(isRenderMathMLRow());
}

RenderMathMLRow::~RenderMathMLRow() = default;

MathMLRowElement& RenderMathMLRow::element() const
{
    return static_cast<MathMLRowElement&>(nodeForNonAnonymous());
}

std::optional<LayoutUnit> RenderMathMLRow::firstLineBaseline() const
{
    auto* baselineChild = firstInFlowChildBox();
    if (!baselineChild)
        return std::optional<LayoutUnit>();

    return LayoutUnit { static_cast<int>(lroundf(ascentForChild(*baselineChild) + baselineChild->marginBefore() + baselineChild->logicalTop())) };
}

static RenderMathMLOperator* toVerticalStretchyOperator(RenderBox* box)
{
    if (auto* mathMLBlock = dynamicDowncast<RenderMathMLBlock>(box)) {
        auto* renderOperator = mathMLBlock->unembellishedOperator();
        if (renderOperator && renderOperator->isStretchy() && renderOperator->isVertical())
            return renderOperator;
    }
    return nullptr;
}

void RenderMathMLRow::stretchVerticalOperatorsAndLayoutChildren()
{
    // First calculate stretch ascent and descent.
    LayoutUnit stretchAscent, stretchDescent;
    for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox()) {
        if (toVerticalStretchyOperator(child))
            continue;
        child->layoutIfNeeded();
        LayoutUnit childAscent = ascentForChild(*child) + child->marginBefore();
        LayoutUnit childDescent = child->logicalHeight() + child->marginLogicalHeight() - childAscent;
        stretchAscent = std::max(stretchAscent, childAscent);
        stretchDescent = std::max(stretchDescent, childDescent);
    }
    if (stretchAscent + stretchDescent <= 0) {
        // We ensure a minimal stretch size.
        stretchAscent = style().computedFontSize();
        stretchDescent = 0;
    }

    // Next, we stretch the vertical operators.
    for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox()) {
        if (auto renderOperator = toVerticalStretchyOperator(child)) {
            renderOperator->stretchTo(stretchAscent, stretchDescent);
            renderOperator->layoutIfNeeded();
            child->layoutIfNeeded();
        }
    }
}

void RenderMathMLRow::getContentBoundingBox(LayoutUnit& width, LayoutUnit& ascent, LayoutUnit& descent) const
{
    ascent = 0;
    descent = 0;
    width = 0;
    for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox()) {
        width += child->marginStart() + child->logicalWidth() + child->marginEnd();
        LayoutUnit childAscent = ascentForChild(*child) + child->marginBefore();
        LayoutUnit childDescent = child->logicalHeight() + child->marginLogicalHeight() - childAscent;
        ascent = std::max(ascent, childAscent);
        descent = std::max(descent, childDescent);
    }
}

LayoutUnit RenderMathMLRow::preferredLogicalWidthOfRowItems()
{
    LayoutUnit preferredWidth = 0;
    for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox()) {
        preferredWidth += child->maxPreferredLogicalWidth() + marginIntrinsicLogicalWidthForChild(*child);
    }
    return preferredWidth;
}

void RenderMathMLRow::computePreferredLogicalWidths()
{
    ASSERT(preferredLogicalWidthsDirty());

    m_minPreferredLogicalWidth = m_maxPreferredLogicalWidth = preferredLogicalWidthOfRowItems();

    auto sizes = sizeAppliedToMathContent(LayoutPhase::CalculatePreferredLogicalWidth);
    applySizeToMathContent(LayoutPhase::CalculatePreferredLogicalWidth, sizes);

    adjustPreferredLogicalWidthsForBorderAndPadding();

    setPreferredLogicalWidthsDirty(false);
}

void RenderMathMLRow::layoutRowItems(LayoutUnit width, LayoutUnit ascent)
{
    LayoutUnit horizontalOffset = 0;
    for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox()) {
        horizontalOffset += child->marginStart();
        LayoutUnit childVerticalOffset = ascent - ascentForChild(*child);
        LayoutUnit childWidth = child->logicalWidth();
        LayoutUnit childHorizontalOffset = writingMode().isBidiLTR() ? horizontalOffset : width - horizontalOffset - childWidth;
        auto repaintRect = child->checkForRepaintDuringLayout() ? std::make_optional(child->frameRect()) : std::nullopt;
        child->setLocation(LayoutPoint(childHorizontalOffset, childVerticalOffset));
        if (repaintRect) {
            repaintRect->uniteEvenIfEmpty(child->frameRect());
            repaintRectangle(*repaintRect);
        }
        horizontalOffset += childWidth + child->marginEnd();
    }
}

void RenderMathMLRow::layoutBlock(bool relayoutChildren, LayoutUnit)
{
    ASSERT(needsLayout());

    insertPositionedChildrenIntoContainingBlock();

    if (!relayoutChildren && simplifiedLayout())
        return;

    layoutFloatingChildren();

    recomputeLogicalWidth();
    computeAndSetBlockDirectionMarginsOfChildren();

    LayoutUnit width, ascent, descent;
    stretchVerticalOperatorsAndLayoutChildren();
    getContentBoundingBox(width, ascent, descent);
    layoutRowItems(width, ascent);
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

}

#endif // ENABLE(MATHML)
