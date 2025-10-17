/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#include "GridLayoutFunctions.h"

#include "AncestorSubgridIterator.h"
#include "LengthFunctions.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderChildIterator.h"
#include "RenderGrid.h"
#include "RenderStyleConstants.h"
#include "RenderStyleInlines.h"

namespace WebCore {

namespace GridLayoutFunctions {

static inline bool marginStartIsAuto(const RenderBox& gridItem, GridTrackSizingDirection direction)
{
    return direction == GridTrackSizingDirection::ForColumns ? gridItem.style().marginStart().isAuto() : gridItem.style().marginBefore().isAuto();
}

static inline bool marginEndIsAuto(const RenderBox& gridItem, GridTrackSizingDirection direction)
{
    return direction == GridTrackSizingDirection::ForColumns ? gridItem.style().marginEnd().isAuto() : gridItem.style().marginAfter().isAuto();
}

static bool gridItemHasMargin(const RenderBox& gridItem, GridTrackSizingDirection direction)
{
    // Length::IsZero returns true for 'auto' margins, which is aligned with the purpose of this function.
    if (direction == GridTrackSizingDirection::ForColumns)
        return !gridItem.style().marginStart().isZero() || !gridItem.style().marginEnd().isZero();
    return !gridItem.style().marginBefore().isZero() || !gridItem.style().marginAfter().isZero();
}

LayoutUnit computeMarginLogicalSizeForGridItem(const RenderGrid& grid, GridTrackSizingDirection direction, const RenderBox& gridItem)
{
    GridTrackSizingDirection flowAwareDirection = flowAwareDirectionForGridItem(grid, gridItem, direction);
    if (!gridItemHasMargin(gridItem, flowAwareDirection))
        return 0;

    LayoutUnit marginStart;
    LayoutUnit marginEnd;
    if (direction == GridTrackSizingDirection::ForColumns)
        gridItem.computeInlineDirectionMargins(grid, gridItem.containingBlockLogicalWidthForContent(), { }, gridItem.logicalWidth(), marginStart, marginEnd);
    else
        gridItem.computeBlockDirectionMargins(grid, marginStart, marginEnd);
    return marginStartIsAuto(gridItem, flowAwareDirection) ? marginEnd : marginEndIsAuto(gridItem, flowAwareDirection) ? marginStart : marginStart + marginEnd;
}

bool hasRelativeOrIntrinsicSizeForGridItem(const RenderBox& gridItem, GridTrackSizingDirection direction)
{
    if (direction == GridTrackSizingDirection::ForColumns)
        return gridItem.hasRelativeLogicalWidth() || gridItem.style().logicalWidth().isIntrinsicOrAuto();
    return gridItem.hasRelativeLogicalHeight() || gridItem.style().logicalHeight().isIntrinsicOrAuto();
}

static ExtraMarginsFromSubgrids extraMarginForSubgrid(const RenderGrid& parent, unsigned startLine, unsigned endLine, GridTrackSizingDirection direction)
{
    unsigned numTracks = parent.numTracks(direction);
    if (!numTracks || !parent.isSubgrid(direction))
        return { };

    std::optional<LayoutUnit> availableSpace;
    if (!hasRelativeOrIntrinsicSizeForGridItem(parent, direction))
        availableSpace = parent.availableSpaceForGutters(direction);

    RenderGrid& grandParent = downcast<RenderGrid>(*parent.parent());
    ExtraMarginsFromSubgrids extraMargins;
    if (!startLine)
        extraMargins.addTrackStartMargin((direction == GridTrackSizingDirection::ForColumns) ? parent.marginAndBorderAndPaddingStart() : parent.marginAndBorderAndPaddingBefore());
    else
        extraMargins.addTrackStartMargin((parent.gridGap(direction, availableSpace) - grandParent.gridGap(direction)) / 2);

    if (endLine == numTracks)
        extraMargins.addTrackEndMargin((direction == GridTrackSizingDirection::ForColumns) ? parent.marginAndBorderAndPaddingEnd() : parent.marginAndBorderAndPaddingAfter());
    else
        extraMargins.addTrackEndMargin((parent.gridGap(direction, availableSpace) - grandParent.gridGap(direction)) / 2);

    return extraMargins;
}

ExtraMarginsFromSubgrids extraMarginForSubgridAncestors(GridTrackSizingDirection direction, const RenderBox& gridItem)
{
    ExtraMarginsFromSubgrids extraMargins;
    for (auto& currentAncestorSubgrid : ancestorSubgridsOfGridItem(gridItem, direction)) {
        GridSpan span = currentAncestorSubgrid.gridSpanForGridItem(gridItem, direction);
        extraMargins += extraMarginForSubgrid(currentAncestorSubgrid, span.startLine(), span.endLine(), direction);
    }
    return extraMargins;
}

LayoutUnit marginLogicalSizeForGridItem(const RenderGrid& grid, GridTrackSizingDirection direction, const RenderBox& gridItem)
{
    auto margin = computeMarginLogicalSizeForGridItem(grid, direction, gridItem);

    if (&grid != gridItem.parent()) {
        GridTrackSizingDirection subgridDirection = flowAwareDirectionForGridItem(grid, *downcast<RenderGrid>(gridItem.parent()), direction);
        margin += extraMarginForSubgridAncestors(subgridDirection, gridItem).extraTotalMargin();
    }

    return margin;
}

bool isOrthogonalGridItem(const RenderGrid& grid, const RenderBox& gridItem)
{
    return gridItem.isHorizontalWritingMode() != grid.isHorizontalWritingMode();
}

bool isOrthogonalParent(const RenderGrid& grid, const RenderElement& parent)
{
    return parent.isHorizontalWritingMode() != grid.isHorizontalWritingMode();
}

bool isAspectRatioBlockSizeDependentGridItem(const RenderBox& gridItem)
{
    return (gridItem.style().hasAspectRatio() || gridItem.hasIntrinsicAspectRatio()) && (gridItem.hasRelativeLogicalHeight() || gridItem.hasStretchedLogicalHeight());
}

bool isGridItemInlineSizeDependentOnBlockConstraints(const RenderBox& gridItem, const RenderGrid& parentGrid, ItemPosition gridItemAlignSelf)
{
    ASSERT(gridItem.parent() == &parentGrid);

    if (isOrthogonalGridItem(parentGrid, gridItem))
        return true;

    auto& gridItemStyle = gridItem.style();
    auto gridItemFlexWrap = gridItemStyle.flexWrap();
    if (gridItem.isRenderFlexibleBox() && gridItem.style().isColumnFlexDirection() && (gridItemFlexWrap == FlexWrap::Wrap || gridItemFlexWrap == FlexWrap::Reverse))
        return true;

    if (gridItem.isRenderMultiColumnFlow())
        return true;

    if (isAspectRatioBlockSizeDependentGridItem(gridItem))
        return true;


    auto hasAspectRatioAndInlineSizeDependsOnBlockSize = [](const RenderObject& renderer) {
        auto& rendererStyle = renderer.style();
        bool rendererHasAspectRatio = renderer.hasIntrinsicAspectRatio() || rendererStyle.hasAspectRatio();

        return rendererHasAspectRatio && rendererStyle.logicalWidth().isAuto() && !rendererStyle.logicalHeight().isIntrinsicOrAuto();
    };

    // Stretch alignment allows the grid item content to resolve against the stretched size.
    if (gridItemAlignSelf != ItemPosition::Stretch)
        return false;

    for (auto& gridItemChild : childrenOfType<RenderObject>(gridItem)) {
        if (hasAspectRatioAndInlineSizeDependsOnBlockSize(gridItemChild))
            return true;
    }

    return false;
}

GridTrackSizingDirection flowAwareDirectionForGridItem(const RenderGrid& grid, const RenderBox& gridItem, GridTrackSizingDirection direction)
{
    return !isOrthogonalGridItem(grid, gridItem) ? direction : (direction == GridTrackSizingDirection::ForColumns ? GridTrackSizingDirection::ForRows : GridTrackSizingDirection::ForColumns);
}

GridTrackSizingDirection flowAwareDirectionForParent(const RenderGrid& grid, const RenderElement& parent, GridTrackSizingDirection direction)
{
    return isOrthogonalParent(grid, parent) ? (direction == GridTrackSizingDirection::ForColumns ? GridTrackSizingDirection::ForRows : GridTrackSizingDirection::ForColumns) : direction;
}

std::optional<RenderBox::GridAreaSize> overridingContainingBlockContentSizeForGridItem(const RenderBox& gridItem, GridTrackSizingDirection direction)
{
    return direction == GridTrackSizingDirection::ForColumns ? gridItem.gridAreaContentLogicalWidth() : gridItem.gridAreaContentLogicalHeight();
}

bool isFlippedDirection(const RenderGrid& grid, GridTrackSizingDirection direction)
{
    if (direction == GridTrackSizingDirection::ForColumns)
        return grid.writingMode().isBidiRTL();
    return grid.writingMode().isBlockFlipped();
}

bool isSubgridReversedDirection(const RenderGrid& grid, GridTrackSizingDirection outerDirection, const RenderGrid& subgrid)
{
    GridTrackSizingDirection subgridDirection = flowAwareDirectionForGridItem(grid, subgrid, outerDirection);
    ASSERT(subgrid.isSubgrid(subgridDirection));
    return isFlippedDirection(grid, outerDirection) != isFlippedDirection(subgrid, subgridDirection);
}

unsigned alignmentContextForBaselineAlignment(const GridSpan& span, const ItemPosition& alignment)
{
    ASSERT(alignment == ItemPosition::Baseline || alignment == ItemPosition::LastBaseline);
    if (alignment == ItemPosition::Baseline)
        return span.startLine();
    return span.endLine() - 1;
}

void setOverridingContentSizeForGridItem(const RenderGrid& renderGrid, RenderBox& gridItem, LayoutUnit logicalSize, GridTrackSizingDirection direction)
{
    if (!isOrthogonalGridItem(renderGrid, gridItem))
        direction == GridTrackSizingDirection::ForColumns ? gridItem.setOverridingBorderBoxLogicalWidth(logicalSize) : gridItem.setOverridingBorderBoxLogicalHeight(logicalSize);
    else
        direction == GridTrackSizingDirection::ForColumns ? gridItem.setOverridingBorderBoxLogicalHeight(logicalSize) : gridItem.setOverridingBorderBoxLogicalWidth(logicalSize);
}

void clearOverridingContentSizeForGridItem(const RenderGrid& renderGrid, RenderBox &gridItem, GridTrackSizingDirection direction)
{
    if (!isOrthogonalGridItem(renderGrid, gridItem))
        direction == GridTrackSizingDirection::ForColumns ? gridItem.clearOverridingBorderBoxLogicalWidth() : gridItem.clearOverridingBorderBoxLogicalHeight();
    else
        direction == GridTrackSizingDirection::ForColumns ? gridItem.clearOverridingBorderBoxLogicalHeight() : gridItem.clearOverridingBorderBoxLogicalWidth();
}


GridAxis gridAxisForDirection(GridTrackSizingDirection direction)
{
    return direction == GridTrackSizingDirection::ForColumns ? GridAxis::GridRowAxis : GridAxis::GridColumnAxis;
}

GridTrackSizingDirection gridDirectionForAxis(GridAxis axis)
{
    return axis == GridAxis::GridRowAxis ? GridTrackSizingDirection::ForColumns : GridTrackSizingDirection::ForRows;
}

} // namespace GridLayoutFunctions

} // namespace WebCore
