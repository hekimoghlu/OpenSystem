/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

#include "LayoutUnit.h"
#include "RenderFlexibleBox.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>

namespace WebCore {

class RenderBox;

class FlexLayoutItem {
public:
    FlexLayoutItem(RenderBox&, LayoutUnit flexBaseContentSize, LayoutUnit mainAxisBorderAndPadding, LayoutUnit mainAxisMargin, std::pair<LayoutUnit, LayoutUnit> minMaxSizes, bool everHadLayout);

    LayoutUnit hypotheticalMainAxisMarginBoxSize() const
    {
        return hypotheticalMainContentSize + mainAxisBorderAndPadding + mainAxisMargin;
    }

    LayoutUnit flexBaseMarginBoxSize() const
    {
        return flexBaseContentSize + mainAxisBorderAndPadding + mainAxisMargin;
    }

    LayoutUnit flexedMarginBoxSize() const
    {
        return flexedContentSize + mainAxisBorderAndPadding + mainAxisMargin;
    }

    const RenderStyle& style() const { return renderer->style(); }

    LayoutUnit constrainSizeByMinMax(const LayoutUnit) const;

    CheckedRef<RenderBox> renderer;
    LayoutUnit flexBaseContentSize;
    const LayoutUnit mainAxisBorderAndPadding;
    mutable LayoutUnit mainAxisMargin;
    const std::pair<LayoutUnit, LayoutUnit> minMaxSizes;
    const LayoutUnit hypotheticalMainContentSize;
    LayoutUnit flexedContentSize;
    bool frozen { false };
    bool everHadLayout { false };
};

class FlexLayoutAlgorithm {
    WTF_MAKE_NONCOPYABLE(FlexLayoutAlgorithm);

public:
    FlexLayoutAlgorithm(RenderFlexibleBox&, LayoutUnit lineBreakLength, const Vector<FlexLayoutItem>& allItems, LayoutUnit gapBetweenItems, LayoutUnit gapBetweenLines);

    // The hypothetical main size of an item is the flex base size clamped
    // according to its min and max main size properties
    bool computeNextFlexLine(size_t& nextIndex, Vector<FlexLayoutItem>& lineItems, LayoutUnit& sumFlexBaseSize, double& totalFlexGrow, double& totalFlexShrink, double& totalWeightedFlexShrink, LayoutUnit& sumHypotheticalMainSize);

private:
    bool isMultiline() const { return m_flexbox.style().flexWrap() != FlexWrap::NoWrap; }
    bool canFitItemWithTrimmedMarginEnd(const FlexLayoutItem&, LayoutUnit sumHypotheticalMainSize) const;
    void removeMarginEndFromFlexSizes(FlexLayoutItem&, LayoutUnit& sumFlexBaseSize, LayoutUnit& sumHypotheticalMainSize) const;

    RenderFlexibleBox& m_flexbox;
    LayoutUnit m_lineBreakLength;
    const Vector<FlexLayoutItem>& m_allItems;

    const LayoutUnit m_gapBetweenItems;
    const LayoutUnit m_gapBetweenLines;
};

} // namespace WebCore

