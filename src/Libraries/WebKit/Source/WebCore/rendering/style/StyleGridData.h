/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
#include "GridTrackSize.h"
#include "RenderStyleConstants.h"
#include "StyleContentAlignmentData.h"
#include <variant>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct NamedGridLinesMap {
    UncheckedKeyHashMap<String, Vector<unsigned>> map;

    friend bool operator==(const NamedGridLinesMap&, const NamedGridLinesMap&) = default;
};

struct OrderedNamedGridLinesMap {
    UncheckedKeyHashMap<unsigned, Vector<String>, IntHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> map;
};

typedef std::variant<GridTrackSize, Vector<String>> RepeatEntry;
typedef Vector<RepeatEntry> RepeatTrackList;

struct GridTrackEntrySubgrid {
    friend bool operator==(const GridTrackEntrySubgrid&, const GridTrackEntrySubgrid&) = default;
};

struct GridTrackEntryMasonry {
    friend bool operator==(const GridTrackEntryMasonry&, const GridTrackEntryMasonry&) = default;
};

struct GridTrackEntryRepeat {
    friend bool operator==(const GridTrackEntryRepeat&, const GridTrackEntryRepeat&) = default;

    unsigned repeats;
    RepeatTrackList list;
};

struct GridTrackEntryAutoRepeat {
    friend bool operator==(const GridTrackEntryAutoRepeat&, const GridTrackEntryAutoRepeat&) = default;

    AutoRepeatType type;
    RepeatTrackList list;
};

struct MasonryAutoFlow {
    friend bool operator==(const MasonryAutoFlow&, const MasonryAutoFlow&) = default;

    MasonryAutoFlowPlacementAlgorithm placementAlgorithm;
    MasonryAutoFlowPlacementOrder placementOrder;
};

using GridTrackEntry = std::variant<GridTrackSize, Vector<String>, GridTrackEntryRepeat, GridTrackEntryAutoRepeat, GridTrackEntrySubgrid, GridTrackEntryMasonry>;
struct GridTrackList {
    Vector<GridTrackEntry> list;
    friend bool operator==(const GridTrackList&, const GridTrackList&) = default;
};
inline WTF::TextStream& operator<<(WTF::TextStream& stream, const GridTrackList& list) { return stream << list.list; }

WTF::TextStream& operator<<(WTF::TextStream&, const RepeatEntry&);
WTF::TextStream& operator<<(WTF::TextStream&, const GridTrackEntry&);

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleGridData);
class StyleGridData : public RefCounted<StyleGridData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleGridData);
public:
    static Ref<StyleGridData> create() { return adoptRef(*new StyleGridData); }
    Ref<StyleGridData> copy() const;

    bool operator==(const StyleGridData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleGridData&) const;
#endif

    void setRows(const GridTrackList&);
    void setColumns(const GridTrackList&);

    const Vector<GridTrackSize>& gridColumnTrackSizes() const { return m_gridColumnTrackSizes; }
    const Vector<GridTrackSize>& gridRowTrackSizes() const { return m_gridRowTrackSizes; }

    const NamedGridLinesMap& namedGridColumnLines() const { return m_namedGridColumnLines; };
    const NamedGridLinesMap& namedGridRowLines() const { return m_namedGridRowLines; };

    const OrderedNamedGridLinesMap& orderedNamedGridColumnLines() const { return m_orderedNamedGridColumnLines; }
    const OrderedNamedGridLinesMap& orderedNamedGridRowLines() const { return m_orderedNamedGridRowLines; }

    const NamedGridLinesMap& autoRepeatNamedGridColumnLines() const { return m_autoRepeatNamedGridColumnLines; }
    const NamedGridLinesMap& autoRepeatNamedGridRowLines() const { return m_autoRepeatNamedGridRowLines; }
    const OrderedNamedGridLinesMap& autoRepeatOrderedNamedGridColumnLines() const { return m_autoRepeatOrderedNamedGridColumnLines; }
    const OrderedNamedGridLinesMap& autoRepeatOrderedNamedGridRowLines() const { return m_autoRepeatOrderedNamedGridRowLines; }

    const Vector<GridTrackSize>& gridAutoRepeatColumns() const { return m_gridAutoRepeatColumns; }
    const Vector<GridTrackSize>& gridAutoRepeatRows() const { return m_gridAutoRepeatRows; }

    const unsigned& autoRepeatColumnsInsertionPoint() const { return m_autoRepeatColumnsInsertionPoint; }
    const unsigned& autoRepeatRowsInsertionPoint() const { return m_autoRepeatRowsInsertionPoint; }

    const AutoRepeatType& autoRepeatColumnsType() const { return m_autoRepeatColumnsType; }
    const AutoRepeatType& autoRepeatRowsType() const { return m_autoRepeatRowsType; }

    const bool& subgridRows() const { return m_subgridRows; };
    const bool& subgridColumns() const { return m_subgridColumns; }

    bool masonryRows() const { return m_masonryRows; }
    bool masonryColumns() const { return m_masonryColumns; }

    const GridTrackList& columns() const { return m_columns; }
    const GridTrackList& rows() const { return m_rows; }

    NamedGridLinesMap implicitNamedGridColumnLines;
    NamedGridLinesMap implicitNamedGridRowLines;

    unsigned gridAutoFlow : GridAutoFlowBits;
    MasonryAutoFlow masonryAutoFlow;

    Vector<GridTrackSize> gridAutoRows;
    Vector<GridTrackSize> gridAutoColumns;

    NamedGridAreaMap namedGridArea;
    // Because namedGridArea doesn't store the unnamed grid areas, we need to keep track
    // of the explicit grid size defined by both named and unnamed grid areas.
    unsigned namedGridAreaRowCount;
    unsigned namedGridAreaColumnCount;

private:
    void computeCachedTrackData(const GridTrackList&, Vector<GridTrackSize>& sizes, NamedGridLinesMap& namedLines, OrderedNamedGridLinesMap& orderedNamedLines, Vector<GridTrackSize>& autoRepeatSizes, NamedGridLinesMap& autoRepeatNamedLines, OrderedNamedGridLinesMap& autoRepeatOrderedNamedLines, unsigned& autoRepeatInsertionPoint, AutoRepeatType&, bool& subgrid, bool& masonry);

    GridTrackList m_columns;
    GridTrackList m_rows;

    // Grid track sizes are computed from m_columns/m_rows.
    Vector<GridTrackSize> m_gridColumnTrackSizes;
    Vector<GridTrackSize> m_gridRowTrackSizes;

    NamedGridLinesMap m_namedGridColumnLines;
    NamedGridLinesMap m_namedGridRowLines;

    OrderedNamedGridLinesMap m_orderedNamedGridColumnLines;
    OrderedNamedGridLinesMap m_orderedNamedGridRowLines;

    NamedGridLinesMap m_autoRepeatNamedGridColumnLines;
    NamedGridLinesMap m_autoRepeatNamedGridRowLines;
    OrderedNamedGridLinesMap m_autoRepeatOrderedNamedGridColumnLines;
    OrderedNamedGridLinesMap m_autoRepeatOrderedNamedGridRowLines;

    Vector<GridTrackSize> m_gridAutoRepeatColumns;
    Vector<GridTrackSize> m_gridAutoRepeatRows;

    unsigned m_autoRepeatColumnsInsertionPoint;
    unsigned m_autoRepeatRowsInsertionPoint;

    AutoRepeatType m_autoRepeatColumnsType;
    AutoRepeatType m_autoRepeatRowsType;

    bool m_subgridRows;
    bool m_subgridColumns;

    bool m_masonryRows;
    bool m_masonryColumns;

    StyleGridData();
    StyleGridData(const StyleGridData&);
};

} // namespace WebCore

