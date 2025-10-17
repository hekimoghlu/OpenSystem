/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#include "RenderMathMLMath.h"

#if ENABLE(MATHML)

#include "MathMLNames.h"
#include "MathMLRowElement.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace MathMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLMath);

RenderMathMLMath::RenderMathMLMath(MathMLRowElement& element, RenderStyle&& style)
    : RenderMathMLRow(Type::MathMLMath, element, WTFMove(style))
{
    ASSERT(isRenderMathMLMath());
}

RenderMathMLMath::~RenderMathMLMath() = default;

void RenderMathMLMath::centerChildren(LayoutUnit contentWidth)
{
    auto centerBlockOffset = (logicalWidth() - contentWidth) / 2;
    if (!centerBlockOffset)
        return;

    if (!style().isLeftToRightDirection())
        centerBlockOffset = -centerBlockOffset;
    for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox()) {
        auto repaintRect = child->checkForRepaintDuringLayout() ? std::make_optional(child->frameRect()) : std::nullopt;
        child->move(centerBlockOffset, { });
        if (repaintRect) {
            repaintRect->uniteEvenIfEmpty(child->frameRect());
            repaintRectangle(*repaintRect);
        }
    }
}

void RenderMathMLMath::layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight)
{
    ASSERT(needsLayout());

    if (style().display() != DisplayType::Block) {
        RenderMathMLRow::layoutBlock(relayoutChildren, pageLogicalHeight);
        return;
    }

    insertPositionedChildrenIntoContainingBlock();

    if (!relayoutChildren && simplifiedLayout())
        return;

    layoutFloatingChildren();

    recomputeLogicalWidth();
    computeAndSetBlockDirectionMarginsOfChildren();

    setLogicalHeight(borderAndPaddingLogicalHeight() + scrollbarLogicalHeight());

    LayoutUnit width, ascent, descent;
    stretchVerticalOperatorsAndLayoutChildren();
    getContentBoundingBox(width, ascent, descent);
    layoutRowItems(logicalWidth(), ascent);

    // Display formulas must be centered horizontally if there is extra space left.
    // Otherwise, logical width must be updated to the content width to avoid truncation.
    if (width < logicalWidth())
        centerChildren(width);
    else
        setLogicalWidth(width);
    shiftInFlowChildren(0_lu, borderAndPaddingBefore());

    setLogicalHeight(ascent + descent + borderAndPaddingLogicalHeight() + horizontalScrollbarHeight());
    updateLogicalHeight();

    layoutPositionedObjects(relayoutChildren);

    updateScrollInfoAfterLayout();

    clearNeedsLayout();
}

}

#endif // ENABLE(MATHML)
