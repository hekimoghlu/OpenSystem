/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "FlexFormattingContext.h"

#include "FlexFormattingUtils.h"
#include "FlexRect.h"
#include "InlineRect.h"
#include "LayoutBoxGeometry.h"
#include "LayoutChildIterator.h"
#include "LayoutContext.h"
#include "LayoutState.h"
#include "LengthFunctions.h"
#include "RenderStyleInlines.h"
#include <wtf/FixedVector.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FlexFormattingContext);

FlexFormattingContext::FlexFormattingContext(const ElementBox& flexBox, LayoutState& globalLayoutState)
    : m_flexBox(flexBox)
    , m_globalLayoutState(globalLayoutState)
    , m_flexFormattingUtils(*this)
    , m_integrationUtils(globalLayoutState)
{
}

void FlexFormattingContext::layout(const ConstraintsForFlexContent& constraints)
{
    auto logicalFlexItems = convertFlexItemsToLogicalSpace(constraints);
    auto flexItemRects = FlexLayout { *this }.layout(constraints, logicalFlexItems);
    setFlexItemsGeometry(logicalFlexItems, flexItemRects, constraints);
    positionOutOfFlowChildren();
}

IntrinsicWidthConstraints FlexFormattingContext::computedIntrinsicWidthConstraints()
{
    return { };
}

FlexLayout::LogicalFlexItems FlexFormattingContext::convertFlexItemsToLogicalSpace(const ConstraintsForFlexContent& constraints)
{
    struct FlexItem {
        LogicalFlexItem::MainAxisGeometry mainAxis;
        LogicalFlexItem::CrossAxisGeometry crossAxis;
        int logicalOrder { 0 };
        CheckedPtr<const ElementBox> layoutBox;
    };

    Vector<FlexItem> flexItemList;
    auto flexItemsNeedReordering = false;

    auto convertVisualToLogical = [&] {
        auto direction = root().style().flexDirection();
        auto previousLogicalOrder = std::optional<int> { };
        auto isMainAxisParallelWithInlineAxis = FlexFormattingUtils::isMainAxisParallelWithInlineAxis(root());
        auto isReversedInCrossAxis = FlexFormattingUtils::areFlexLinesReversedInCrossAxis(root());

        for (auto* flexItem = root().firstInFlowChild(); flexItem; flexItem = flexItem->nextInFlowSibling()) {
            auto& flexItemGeometry = m_globalLayoutState.geometryForBox(*flexItem);
            auto& style = flexItem->style();
            auto mainAxis = LogicalFlexItem::MainAxisGeometry { };
            auto crossAxis = LogicalFlexItem::CrossAxisGeometry { };

            auto propertyValueForLength = [&](auto& propertyValue, auto availableSize) -> std::optional<LayoutUnit> {
                if (propertyValue.isFixed())
                    return LayoutUnit { propertyValue.value() };
                if (propertyValue.isSpecified() && availableSize)
                    return valueForLength(propertyValue, *availableSize);
                return { };
            };

            auto setMainAxisValues = [&] {
                auto flexContainerMainInnerSize = constraints.mainAxis().availableSize;

                mainAxis.size = propertyValueForLength(isMainAxisParallelWithInlineAxis ? style.width() : style.height(), flexContainerMainInnerSize);
                mainAxis.minimumSize = propertyValueForLength(isMainAxisParallelWithInlineAxis ? style.minWidth() : style.minHeight(), flexContainerMainInnerSize);
                mainAxis.maximumSize = propertyValueForLength(isMainAxisParallelWithInlineAxis ? style.maxWidth() : style.maxHeight(), flexContainerMainInnerSize);
                // Auto keyword retrieves the value of the main size property as the used flex-basis.
                // If that value is itself auto, then the used value is content.
                mainAxis.definiteFlexBasis = style.flexBasis().isAuto() ? mainAxis.size : propertyValueForLength(style.flexBasis(), flexContainerMainInnerSize);

                auto marginStart = [&]() -> std::optional<LayoutUnit> {
                    if (direction == FlexDirection::Row && !style.marginStart().isAuto())
                        return flexItemGeometry.marginStart();
                    if (direction == FlexDirection::RowReverse && !style.marginEnd().isAuto())
                        return flexItemGeometry.marginEnd();
                    if (direction == FlexDirection::Column && !style.marginBefore().isAuto())
                        return flexItemGeometry.marginBefore();
                    if (direction == FlexDirection::ColumnReverse && !style.marginAfter().isAuto())
                        return flexItemGeometry.marginAfter();
                    return { };
                };
                auto marginEnd = [&]() -> std::optional<LayoutUnit> {
                    if (direction == FlexDirection::Row && !style.marginEnd().isAuto())
                        return flexItemGeometry.marginEnd();
                    if (direction == FlexDirection::RowReverse && !style.marginStart().isAuto())
                        return flexItemGeometry.marginStart();
                    if (direction == FlexDirection::Column && !style.marginAfter().isAuto())
                        return flexItemGeometry.marginAfter();
                    if (direction == FlexDirection::ColumnReverse && !style.marginBefore().isAuto())
                        return flexItemGeometry.marginBefore();
                    return { };
                };
                auto shouldFlipMargins = isMainAxisParallelWithInlineAxis && root().writingMode().isLineInverted();
                mainAxis.marginStart = !shouldFlipMargins ? marginStart() : marginEnd();
                mainAxis.marginEnd = !shouldFlipMargins ? marginEnd() : marginStart();
                mainAxis.borderAndPadding = isMainAxisParallelWithInlineAxis ? flexItemGeometry.horizontalBorderAndPadding() : flexItemGeometry.verticalBorderAndPadding();
            };
            setMainAxisValues();

            auto setCrossAxisValues = [&] {
                auto flexContainerCrossInnerSize = constraints.crossAxis().availableSize;

                crossAxis.definiteSize = propertyValueForLength(isMainAxisParallelWithInlineAxis ? style.height() : style.width(), flexContainerCrossInnerSize);
                crossAxis.minimumSize = propertyValueForLength(isMainAxisParallelWithInlineAxis ? style.minHeight() : style.minWidth(), flexContainerCrossInnerSize);
                crossAxis.maximumSize = propertyValueForLength(isMainAxisParallelWithInlineAxis ? style.maxHeight() : style.maxWidth(), flexContainerCrossInnerSize);

                auto marginStart = [&]() -> std::optional<LayoutUnit> {
                    if (!isReversedInCrossAxis) {
                        if (direction == FlexDirection::Row || direction == FlexDirection::RowReverse)
                            return !style.marginBefore().isAuto() ? std::make_optional(flexItemGeometry.marginBefore()) : std::nullopt;
                        if (direction == FlexDirection::Column || direction == FlexDirection::ColumnReverse)
                            return !style.marginStart().isAuto() ? std::make_optional(flexItemGeometry.marginStart()) : std::nullopt;
                        return { };
                    }
                    if (direction == FlexDirection::Row || direction == FlexDirection::RowReverse)
                        return !style.marginAfter().isAuto() ? std::make_optional(flexItemGeometry.marginAfter()) : std::nullopt;
                    if (direction == FlexDirection::Column || direction == FlexDirection::ColumnReverse)
                        return !style.marginEnd().isAuto() ? std::make_optional(flexItemGeometry.marginEnd()) : std::nullopt;
                    return { };
                };
                auto marginEnd = [&]() -> std::optional<LayoutUnit> {
                    if (!isReversedInCrossAxis) {
                        if (direction == FlexDirection::Row || direction == FlexDirection::RowReverse)
                            return !style.marginAfter().isAuto() ? std::make_optional(flexItemGeometry.marginAfter()) : std::nullopt;
                        if (direction == FlexDirection::Column || direction == FlexDirection::ColumnReverse)
                            return !style.marginEnd().isAuto() ? std::make_optional(flexItemGeometry.marginEnd()) : std::nullopt;
                        return { };
                    }
                    if (direction == FlexDirection::Row || direction == FlexDirection::RowReverse)
                        return !style.marginBefore().isAuto() ? std::make_optional(flexItemGeometry.marginBefore()) : std::nullopt;
                    if (direction == FlexDirection::Column || direction == FlexDirection::ColumnReverse)
                        return !style.marginStart().isAuto() ? std::make_optional(flexItemGeometry.marginStart()) : std::nullopt;
                    return { };
                };
                auto shouldFlipMargins = !isMainAxisParallelWithInlineAxis && root().writingMode().isLineInverted();
                crossAxis.marginStart = !shouldFlipMargins ? marginStart() : marginEnd();
                crossAxis.marginEnd = !shouldFlipMargins ? marginEnd() : marginStart();

                crossAxis.hasSizeAuto = isMainAxisParallelWithInlineAxis ? style.height().isAuto() : style.width().isAuto();
                crossAxis.borderAndPadding = isMainAxisParallelWithInlineAxis ? flexItemGeometry.verticalBorderAndPadding() : flexItemGeometry.horizontalBorderAndPadding();
            };
            setCrossAxisValues();

            auto flexItemOrder = style.order();
            flexItemsNeedReordering = flexItemsNeedReordering || flexItemOrder != previousLogicalOrder.value_or(0);
            previousLogicalOrder = flexItemOrder;

            flexItemList.append({ mainAxis, crossAxis, flexItemOrder, downcast<ElementBox>(flexItem) });
        }
    };
    convertVisualToLogical();

    auto reorderFlexItemsIfApplicable = [&] {
        if (!flexItemsNeedReordering)
            return;

        std::stable_sort(flexItemList.begin(), flexItemList.end(), [&] (auto& a, auto& b) {
            return a.logicalOrder < b.logicalOrder;
        });
    };
    reorderFlexItemsIfApplicable();

    auto logicalFlexItemList = FlexLayout::LogicalFlexItems(flexItemList.size());
    for (size_t index = 0; index < flexItemList.size(); ++index) {
        auto& flexItem = flexItemList[index];
        logicalFlexItemList[index] = { *flexItem.layoutBox
            , flexItem.mainAxis
            , flexItem.crossAxis
            , false
            , false
        };
    }
    return logicalFlexItemList;
}

void FlexFormattingContext::setFlexItemsGeometry(const FlexLayout::LogicalFlexItems& logicalFlexItemList, const FlexLayout::LogicalFlexItemRects& logicalRects, const ConstraintsForFlexContent& constraints)
{
    auto& flexBoxStyle = root().style();
    auto flexDirection = flexBoxStyle.flexDirection();
    auto isLeftToRightDirection = flexBoxStyle.writingMode().isLogicalLeftInlineStart();
    auto isRowDirection = flexDirection == FlexDirection::Row || flexDirection == FlexDirection::RowReverse;
    auto flexContainerContentBoxPosition = LayoutPoint { isRowDirection ? constraints.mainAxis().startPosition : constraints.crossAxis().startPosition, isRowDirection ? constraints.crossAxis().startPosition : constraints.mainAxis().startPosition };
    auto flexContainerMainAxisSize = [&] {
        if (auto size = constraints.mainAxis().availableSize)
            return *size;
        // Let's use content size when available size is inf.
        auto& lastFlexItem = logicalFlexItemList.last();
        auto& lastRect = logicalRects.last();
        return lastRect.right() + lastRect.marginRight() + (lastFlexItem.isContentBoxBased() ? geometryForFlexItem(lastFlexItem.layoutBox()).horizontalBorderAndPadding() : 0_lu);
    }();
    auto flexContainerCrossAxisSize = [&] {
        if (auto crossAxisSize = constraints.crossAxis().availableSize)
            return *crossAxisSize;
        // In case of content size driven height in reversed cross axis direction (content is upside down), the height is just the farthest point with content.
        auto maximumHeight = LayoutUnit { };
        for (auto& logicalRect : logicalRects)
            maximumHeight = std::max(maximumHeight, logicalRect.bottom() + logicalRect.marginBottom());
        return maximumHeight;
    }();

    for (size_t index = 0; index < logicalFlexItemList.size(); ++index) {
        auto& logicalFlexItem = logicalFlexItemList[index];
        auto& flexItemGeometry = geometryForFlexItem(logicalFlexItem.layoutBox());
        auto logicalRect = [&] {
            // Note that flex rects are inner size based.
            if (flexBoxStyle.flexWrap() != FlexWrap::Reverse)
                return logicalRects[index];
            auto rect = logicalRects[index];
            auto adjustedLogicalBorderBoxTop = flexContainerCrossAxisSize - rect.bottom();
            if (logicalFlexItem.isContentBoxBased())
                adjustedLogicalBorderBoxTop -= flexDirection == FlexDirection::Row || flexDirection == FlexDirection::RowReverse ? flexItemGeometry.verticalBorderAndPadding() : flexItemGeometry.horizontalBorderAndPadding();
            rect.setTop(adjustedLogicalBorderBoxTop);
            return rect;
        }();

        auto borderBoxTop = LayoutUnit { };
        auto borderBoxLeft = LayoutUnit { };
        if (flexDirection == FlexDirection::Row || flexDirection == FlexDirection::RowReverse) {
            borderBoxTop += logicalRect.top();
            if (flexDirection == FlexDirection::Row)
                borderBoxLeft = logicalRect.left();
            else {
                borderBoxLeft = flexContainerMainAxisSize - logicalRect.right();
                if (logicalFlexItem.isContentBoxBased())
                    borderBoxLeft -= flexItemGeometry.horizontalBorderAndPadding();
            }
        } else {
            // Let's flip x and y to go from column to row.
            borderBoxLeft = logicalRect.top();
            if (flexDirection == FlexDirection::Column)
                borderBoxTop = logicalRect.left();
            else {
                borderBoxTop = flexContainerMainAxisSize - logicalRect.right();
                if (logicalFlexItem.isContentBoxBased())
                    borderBoxTop -= flexItemGeometry.verticalBorderAndPadding();
            }
        }
        auto contentBoxWidth = isRowDirection ? logicalRect.width() : logicalRect.height();
        auto contentBoxHeight = isRowDirection ? logicalRect.height() : logicalRect.width();
        if (!logicalFlexItem.isContentBoxBased()) {
            contentBoxWidth -= flexItemGeometry.horizontalBorderAndPadding();
            contentBoxHeight -= flexItemGeometry.verticalBorderAndPadding();
        }
        flexItemGeometry.setContentBoxWidth(contentBoxWidth);
        flexItemGeometry.setContentBoxHeight(contentBoxHeight);

        if (!isLeftToRightDirection)
            borderBoxLeft = (isRowDirection ? flexContainerMainAxisSize : flexContainerCrossAxisSize) - (borderBoxLeft + flexItemGeometry.borderBoxWidth());
        flexItemGeometry.setTopLeft({ flexContainerContentBoxPosition.x() + borderBoxLeft, flexContainerContentBoxPosition.y() + borderBoxTop });
    }
}

void FlexFormattingContext::positionOutOfFlowChildren()
{
    // FIXME: Implement out-of-flow positioning.
    for (auto* outOfFlowChild = root().firstOutOfFlowChild(); outOfFlowChild; outOfFlowChild = outOfFlowChild->nextOutOfFlowSibling())
        m_globalLayoutState.ensureGeometryForBox(*outOfFlowChild).setTopLeft({ });
}

const BoxGeometry& FlexFormattingContext::geometryForFlexItem(const Box& flexItem) const
{
    ASSERT(flexItem.isFlexItem());
    return m_globalLayoutState.geometryForBox(flexItem);
}

BoxGeometry& FlexFormattingContext::geometryForFlexItem(const Box& flexItem)
{
    ASSERT(flexItem.isFlexItem());
    return m_globalLayoutState.ensureGeometryForBox(flexItem);
}


}
}
