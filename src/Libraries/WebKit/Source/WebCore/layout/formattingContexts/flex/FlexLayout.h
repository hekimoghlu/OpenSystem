/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

#include "FlexFormattingConstraints.h"
#include "FlexRect.h"
#include "LogicalFlexItem.h"
#include "RenderStyleConstants.h"
#include <wtf/Range.h>

namespace WebCore {

class RenderStyle;

namespace Layout {

class FlexFormattingContext;
class FlexFormattingUtils;
struct FlexBaseAndHypotheticalMainSize;
struct PositionAndMargins;

// This class implements the layout logic for flex formatting contexts.
// https://www.w3.org/TR/css-flexbox-1/
class FlexLayout {
public:
    FlexLayout(FlexFormattingContext&);

    using LogicalFlexItems = Vector<LogicalFlexItem>;
    using LogicalFlexItemRects = FixedVector<FlexRect>;
    LogicalFlexItemRects layout(const ConstraintsForFlexContent&, const LogicalFlexItems&);

private:
    using FlexBaseAndHypotheticalMainSizeList = Vector<FlexBaseAndHypotheticalMainSize>;
    using LineRanges = Vector<WTF::Range<size_t>>;
    using SizeList = FixedVector<LayoutUnit>;
    using PositionAndMarginsList = FixedVector<PositionAndMargins>;
    using LinesCrossSizeList = Vector<LayoutUnit>;
    using LinesCrossPositionList = Vector<LayoutUnit>;

    FlexBaseAndHypotheticalMainSizeList flexBaseAndHypotheticalMainSizeForFlexItems(const LogicalFlexItems&, bool isSizedUnderMinMaxConstraints) const;
    LayoutUnit flexContainerInnerMainSize(const ConstraintsForFlexContent::AxisGeometry&) const;
    LineRanges computeFlexLines(const LogicalFlexItems&, LayoutUnit flexContainerInnerMainSize, const FlexBaseAndHypotheticalMainSizeList&) const;
    SizeList computeMainSizeForFlexItems(const LogicalFlexItems&, const LineRanges&, LayoutUnit flexContainerInnerMainSize, const FlexBaseAndHypotheticalMainSizeList&) const;
    SizeList hypotheticalCrossSizeForFlexItems(const LogicalFlexItems&, const SizeList& flexItemsMainSizeList);
    LinesCrossSizeList crossSizeForFlexLines(const LineRanges&, const ConstraintsForFlexContent::AxisGeometry& crossAxis, const LogicalFlexItems&, const SizeList& flexItemsHypotheticalCrossSizeList) const;
    void stretchFlexLines(LinesCrossSizeList& flexLinesCrossSizeList, size_t numberOfLines, std::optional<LayoutUnit> crossAxisAvailableSpace) const;
    bool collapseNonVisibleFlexItems();
    SizeList computeCrossSizeForFlexItems(const LogicalFlexItems&, const LineRanges&, const LinesCrossSizeList& flexLinesCrossSizeList, const SizeList& flexItemsHypotheticalCrossSizeList) const;
    PositionAndMarginsList handleMainAxisAlignment(LayoutUnit availableMainSpace, const LineRanges&, const LogicalFlexItems&, const SizeList& flexItemsMainSizeList) const;
    PositionAndMarginsList handleCrossAxisAlignmentForFlexItems(const LogicalFlexItems&, const LineRanges&, const SizeList& flexItemsCrossSizeList, const LinesCrossSizeList& flexLinesCrossSizeList) const;
    LinesCrossPositionList handleCrossAxisAlignmentForFlexLines(std::optional<LayoutUnit> crossAxisAvailableSpace, const LineRanges&, LinesCrossSizeList& flexLinesCrossSizeList) const;

    LayoutUnit mainAxisAvailableSpaceForItemAlignment(LayoutUnit mainAxisAvailableSpace, size_t numberOfFlexItems) const;
    LayoutUnit crossAxisAvailableSpaceForLineSizingAndAlignment(LayoutUnit crossAxisAvailableSpace, size_t numberOfFlexLines) const;

    bool isSingleLineFlexContainer() const { return flexContainer().style().flexWrap() == FlexWrap::NoWrap; }
    const ElementBox& flexContainer() const;
    const RenderStyle& flexContainerStyle() const { return flexContainer().style(); }

    const FlexFormattingContext& formattingContext() const;
    const FlexFormattingUtils& formattingUtils() const;

private:
    FlexFormattingContext& m_flexFormattingContext;
};

}
}
