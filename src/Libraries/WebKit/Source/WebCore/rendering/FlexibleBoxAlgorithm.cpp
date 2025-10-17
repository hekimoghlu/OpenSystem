/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
#include "FlexibleBoxAlgorithm.h"

#include "RenderBox.h"
#include "RenderStyleInlines.h"

namespace WebCore {

FlexLayoutItem::FlexLayoutItem(RenderBox& flexItem, LayoutUnit flexBaseContentSize, LayoutUnit mainAxisBorderAndPadding, LayoutUnit mainAxisMargin, std::pair<LayoutUnit, LayoutUnit> minMaxSizes, bool everHadLayout)
    : renderer(flexItem)
    , flexBaseContentSize(flexBaseContentSize)
    , mainAxisBorderAndPadding(mainAxisBorderAndPadding)
    , mainAxisMargin(mainAxisMargin)
    , minMaxSizes(minMaxSizes)
    , hypotheticalMainContentSize(constrainSizeByMinMax(flexBaseContentSize))
    , frozen(false)
    , everHadLayout(everHadLayout)
{
    ASSERT(!flexItem.isOutOfFlowPositioned());
}

FlexLayoutAlgorithm::FlexLayoutAlgorithm(RenderFlexibleBox& flexbox, LayoutUnit lineBreakLength, const Vector<FlexLayoutItem>& allItems, LayoutUnit gapBetweenItems, LayoutUnit gapBetweenLines)
    : m_flexbox(flexbox)
    , m_lineBreakLength(lineBreakLength)
    , m_allItems(allItems)
    , m_gapBetweenItems(gapBetweenItems)
    , m_gapBetweenLines(gapBetweenLines)
{
}

bool FlexLayoutAlgorithm::canFitItemWithTrimmedMarginEnd(const FlexLayoutItem& flexLayoutItem, LayoutUnit sumHypotheticalMainSize) const
{
    auto marginTrim = m_flexbox.style().marginTrim();
    if ((m_flexbox.isHorizontalFlow() && marginTrim.contains(MarginTrimType::InlineEnd)) || (m_flexbox.isColumnFlow() && marginTrim.contains(MarginTrimType::BlockEnd)))
        return sumHypotheticalMainSize + flexLayoutItem.hypotheticalMainAxisMarginBoxSize() - m_flexbox.flowAwareMarginEndForFlexItem(flexLayoutItem.renderer) <= m_lineBreakLength;
    return false;
}

void FlexLayoutAlgorithm::removeMarginEndFromFlexSizes(FlexLayoutItem& flexLayoutItem, LayoutUnit& sumFlexBaseSize, LayoutUnit& sumHypotheticalMainSize) const
{
    LayoutUnit margin;
    if (m_flexbox.isHorizontalFlow())
        margin = flexLayoutItem.renderer->marginEnd(m_flexbox.writingMode());
    else
        margin = flexLayoutItem.renderer->marginAfter(m_flexbox.writingMode());
    sumFlexBaseSize -= margin;
    sumHypotheticalMainSize -= margin;
} 

bool FlexLayoutAlgorithm::computeNextFlexLine(size_t& nextIndex, Vector<FlexLayoutItem>& lineItems, LayoutUnit& sumFlexBaseSize, double& totalFlexGrow, double& totalFlexShrink, double& totalWeightedFlexShrink, LayoutUnit& sumHypotheticalMainSize)
{
    lineItems.clear();
    sumFlexBaseSize = 0_lu;
    totalFlexGrow = totalFlexShrink = totalWeightedFlexShrink = 0;
    sumHypotheticalMainSize = 0_lu;

    // Trim main axis margin for item at the start of the flex line
    if (nextIndex < m_allItems.size() && m_flexbox.shouldTrimMainAxisMarginStart())
        m_flexbox.trimMainAxisMarginStart(m_allItems[nextIndex]);
    for (; nextIndex < m_allItems.size(); ++nextIndex) {
        const auto& flexLayoutItem = m_allItems[nextIndex];
        auto& style = flexLayoutItem.style();
        ASSERT(!flexLayoutItem.renderer->isOutOfFlowPositioned());
        if (isMultiline() && (sumHypotheticalMainSize + flexLayoutItem.hypotheticalMainAxisMarginBoxSize() > m_lineBreakLength && !canFitItemWithTrimmedMarginEnd(flexLayoutItem, sumHypotheticalMainSize)) && !lineItems.isEmpty())
            break;
        lineItems.append(flexLayoutItem);
        sumFlexBaseSize += flexLayoutItem.flexBaseMarginBoxSize() + m_gapBetweenItems;
        totalFlexGrow += style.flexGrow();
        totalFlexShrink += style.flexShrink();
        totalWeightedFlexShrink += style.flexShrink() * flexLayoutItem.flexBaseContentSize;
        sumHypotheticalMainSize += flexLayoutItem.hypotheticalMainAxisMarginBoxSize() + m_gapBetweenItems;
    }

    if (!lineItems.isEmpty()) {
        // We added a gap after every item but there shouldn't be one after the last item, so subtract it here. Note that
        // sums might be negative here due to negative margins in flex items.
        sumHypotheticalMainSize -= m_gapBetweenItems;
        sumFlexBaseSize -= m_gapBetweenItems;
    }

    ASSERT(lineItems.size() > 0 || nextIndex == m_allItems.size());
    // Trim main axis margin for item at the end of the flex line
    if (lineItems.size() && m_flexbox.shouldTrimMainAxisMarginEnd()) {
        auto lastItem = lineItems.last();
        removeMarginEndFromFlexSizes(lastItem, sumFlexBaseSize, sumHypotheticalMainSize);
        m_flexbox.trimMainAxisMarginEnd(lastItem);
    }
    return lineItems.size() > 0;
}

LayoutUnit FlexLayoutItem::constrainSizeByMinMax(const LayoutUnit size) const
{
    return std::max(minMaxSizes.first, std::min(size, minMaxSizes.second));
}

} // namespace WebCore
