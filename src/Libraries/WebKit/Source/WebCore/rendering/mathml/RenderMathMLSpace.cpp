/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#include "RenderMathMLSpace.h"

#if ENABLE(MATHML)

#include "GraphicsContext.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLSpace);

RenderMathMLSpace::RenderMathMLSpace(MathMLSpaceElement& element, RenderStyle&& style)
    : RenderMathMLBlock(Type::MathMLSpace, element, WTFMove(style))
{
    ASSERT(isRenderMathMLSpace());
}

RenderMathMLSpace::~RenderMathMLSpace() = default;

void RenderMathMLSpace::computePreferredLogicalWidths()
{
    ASSERT(preferredLogicalWidthsDirty());

    m_minPreferredLogicalWidth = m_maxPreferredLogicalWidth = spaceWidth();

    auto sizes = sizeAppliedToMathContent(LayoutPhase::CalculatePreferredLogicalWidth);
    applySizeToMathContent(LayoutPhase::CalculatePreferredLogicalWidth, sizes);

    adjustPreferredLogicalWidthsForBorderAndPadding();

    setPreferredLogicalWidthsDirty(false);
}

LayoutUnit RenderMathMLSpace::spaceWidth() const
{
    Ref spaceElement = element();
    // FIXME: Negative width values are not supported yet.
    return std::max<LayoutUnit>(0, toUserUnits(spaceElement->width(), style(), 0));
}

void RenderMathMLSpace::getSpaceHeightAndDepth(LayoutUnit& height, LayoutUnit& depth) const
{
    Ref spaceElement = element();
    height = toUserUnits(spaceElement->height(), style(), 0);
    depth = toUserUnits(spaceElement->depth(), style(), 0);

    // If the total height is negative, set vertical dimensions to 0.
    if (height + depth < 0) {
        height = 0;
        depth = 0;
    }
}

void RenderMathMLSpace::layoutBlock(bool relayoutChildren, LayoutUnit)
{
    ASSERT(needsLayout());

    insertPositionedChildrenIntoContainingBlock();

    if (!relayoutChildren && simplifiedLayout())
        return;

    layoutFloatingChildren();

    recomputeLogicalWidth();

    setLogicalWidth(spaceWidth());
    LayoutUnit height, depth;
    getSpaceHeightAndDepth(height, depth);
    setLogicalHeight(height + depth);

    auto sizes = sizeAppliedToMathContent(LayoutPhase::Layout);
    applySizeToMathContent(LayoutPhase::Layout, sizes);

    adjustLayoutForBorderAndPadding();

    updateScrollInfoAfterLayout();

    clearNeedsLayout();
}

std::optional<LayoutUnit> RenderMathMLSpace::firstLineBaseline() const
{
    LayoutUnit height, depth;
    getSpaceHeightAndDepth(height, depth);
    return height + borderAndPaddingBefore();
}

}

#endif
