/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#include "GridPositionsResolver.h"
#include "OrderIterator.h"
#include <wtf/HashMap.h>
#include <wtf/ListHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

typedef Vector<SingleThreadWeakPtr<RenderBox>, 1> GridCell;
typedef Vector<Vector<GridCell>> GridAsMatrix;
typedef ListHashSet<size_t> OrderedTrackIndexSet;

class GridArea;
class GridPositionsResolver;
class RenderGrid;

class Grid final {
public:
    explicit Grid(RenderGrid&);

    unsigned numTracks(GridTrackSizingDirection) const;

    void ensureGridSize(unsigned maximumRowSize, unsigned maximumColumnSize);
    GridArea insert(RenderBox&, const GridArea&);

    // Note that each in flow child of a grid container becomes a grid item. This means that
    // this method will return false for a grid container with only out of flow children.
    bool hasGridItems() const { return !m_gridItemArea.isEmpty(); }

    GridArea gridItemArea(const RenderBox& item) const;
    void setGridItemArea(const RenderBox& item, GridArea);

    GridSpan gridItemSpan(const RenderBox&, GridTrackSizingDirection) const;
    GridSpan gridItemSpanIgnoringCollapsedTracks(const RenderBox&, GridTrackSizingDirection) const;

    const GridCell& cell(unsigned row, unsigned column) const;

    unsigned explicitGridStart(GridTrackSizingDirection) const;
    void setExplicitGridStart(unsigned rowStart, unsigned columnStart);

    unsigned autoRepeatTracks(GridTrackSizingDirection) const;
    void setAutoRepeatTracks(unsigned autoRepeatRows, unsigned autoRepeatColumns);

    void setClampingForSubgrid(unsigned maxRows, unsigned maxColumns);

    void clampAreaToSubgridIfNeeded(GridArea&);

    void setAutoRepeatEmptyColumns(std::unique_ptr<OrderedTrackIndexSet>);
    void setAutoRepeatEmptyRows(std::unique_ptr<OrderedTrackIndexSet>);

    unsigned autoRepeatEmptyTracksCount(GridTrackSizingDirection) const;
    bool hasAutoRepeatEmptyTracks(GridTrackSizingDirection) const;
    bool isEmptyAutoRepeatTrack(GridTrackSizingDirection, unsigned) const;

    OrderedTrackIndexSet* autoRepeatEmptyTracks(GridTrackSizingDirection) const;

    OrderIterator& orderIterator() { return m_orderIterator; }

    void setNeedsItemsPlacement(bool);
    bool needsItemsPlacement() const { return m_needsItemsPlacement; };

    void setupGridForMasonryLayout();
    unsigned maxRows() const { return m_maxRows; }
    unsigned maxColumns() const { return m_maxColumns; }
private:
    void ensureStorageForRow(unsigned row);

    OrderIterator m_orderIterator;

    unsigned m_explicitColumnStart { 0 };
    unsigned m_explicitRowStart { 0 };

    unsigned m_autoRepeatColumns { 0 };
    unsigned m_autoRepeatRows { 0 };

    unsigned m_maxColumns { 0 };
    unsigned m_maxRows { 0 };

    bool m_needsItemsPlacement { true };

    GridAsMatrix m_grid;

    UncheckedKeyHashMap<SingleThreadWeakRef<const RenderBox>, GridArea> m_gridItemArea;

    std::unique_ptr<OrderedTrackIndexSet> m_autoRepeatEmptyColumns;
    std::unique_ptr<OrderedTrackIndexSet> m_autoRepeatEmptyRows;
};

class GridIterator {
    WTF_MAKE_NONCOPYABLE(GridIterator);
public:
    // |direction| is the direction that is fixed to |fixedTrackIndex| so e.g
    // GridIterator(m_grid, ForColumns, 1) will walk over the rows of the 2nd column.
    GridIterator(const Grid&, GridTrackSizingDirection, unsigned fixedTrackIndex, unsigned varyingTrackIndex = 0);

    static GridIterator createForSubgrid(const RenderGrid& subgrid, const GridIterator& outer, GridSpan subgridSpanInOuter);

    RenderBox* nextGridItem();
    bool isEmptyAreaEnough(unsigned rowSpan, unsigned columnSpan) const;
    std::optional<GridArea> nextEmptyGridArea(unsigned fixedTrackSpan, unsigned varyingTrackSpan);

    GridTrackSizingDirection direction() const
    {
        return m_direction;
    }

private:
    const Grid& m_grid;
    GridTrackSizingDirection m_direction;
    unsigned m_rowIndex;
    unsigned m_columnIndex;
    unsigned m_gridItemIndex;
};

} // namespace WebCore
