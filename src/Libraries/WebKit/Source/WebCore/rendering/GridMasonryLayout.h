/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

#include "GridArea.h"
#include "GridPositionsResolver.h"
#include "GridTrackSizingAlgorithm.h"
#include "LayoutUnit.h"
#include "RenderBox.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class RenderGrid;

class GridMasonryLayout {
public:
    GridMasonryLayout(RenderGrid& renderGrid)
        : m_renderGrid(renderGrid)
    {
    }

    enum class MasonryLayoutPhase : uint8_t {
        LayoutPhase,
        MinContentPhase,
        MaxContentPhase
    };

    void initializeMasonry(unsigned gridAxisTracks, GridTrackSizingDirection masonryAxisDirection);
    void performMasonryPlacement(const GridTrackSizingAlgorithm&, unsigned gridAxisTracks, GridTrackSizingDirection masonryAxisDirection, GridMasonryLayout::MasonryLayoutPhase);
    LayoutUnit offsetForGridItem(const RenderBox&) const;
    LayoutUnit gridContentSize() const { return m_gridContentSize; };
    LayoutUnit gridGap() const { return m_masonryAxisGridGap; };

private:
    GridSpan gridAxisPositionUsingPackAutoFlow(const RenderBox& item) const;
    GridSpan gridAxisPositionUsingNextAutoFlow(const RenderBox& item);
    GridArea gridAreaForIndefiniteGridAxisItem(const RenderBox& item);
    GridArea gridAreaForDefiniteGridAxisItem(const RenderBox&) const;

    void placeMasonryItems(const GridTrackSizingAlgorithm&, GridMasonryLayout::MasonryLayoutPhase);
    void setItemGridAxisContainingBlockToGridArea(const GridTrackSizingAlgorithm&, RenderBox&);
    void insertIntoGridAndLayoutItem(const GridTrackSizingAlgorithm&, RenderBox&, const GridArea&, GridMasonryLayout::MasonryLayoutPhase);
    LayoutUnit calculateMasonryIntrinsicLogicalWidth(RenderBox&, GridMasonryLayout::MasonryLayoutPhase);

    void resizeAndResetRunningPositions();
    LayoutUnit masonryAxisMarginBoxForItem(const RenderBox& gridItem);
    void updateRunningPositions(const RenderBox& gridItem, const GridArea&);
    void updateItemOffset(const RenderBox& gridItem, LayoutUnit offset);
    inline GridTrackSizingDirection gridAxisDirection() const;

    bool hasDefiniteGridAxisPosition(const RenderBox& gridItem, GridTrackSizingDirection masonryDirection) const;
    GridArea masonryGridAreaFromGridAxisSpan(const GridSpan&) const;
    GridSpan gridAxisSpanFromArea(const GridArea&) const;
    bool hasEnoughSpaceAtPosition(unsigned startingPosition, unsigned spanLength) const;

    unsigned m_gridAxisTracksCount;

    Vector<LayoutUnit> m_runningPositions;
    UncheckedKeyHashMap<SingleThreadWeakRef<const RenderBox>, LayoutUnit> m_itemOffsets;
    RenderGrid& m_renderGrid;
    LayoutUnit m_masonryAxisGridGap;
    LayoutUnit m_gridContentSize;

    GridTrackSizingDirection m_masonryAxisDirection;
    const GridSpan m_masonryAxisSpan = GridSpan::masonryAxisTranslatedDefiniteGridSpan();

    unsigned m_autoFlowNextCursor;
};

} // end namespace WebCore
