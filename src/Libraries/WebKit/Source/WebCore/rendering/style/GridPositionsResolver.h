/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

#include "GridPosition.h"

namespace WebCore {

class GridSpan;
class RenderBox;
class RenderGrid;
class RenderStyle;

enum class GridTrackSizingDirection : uint8_t {
    ForColumns,
    ForRows
};

class NamedLineCollectionBase {
    WTF_MAKE_NONCOPYABLE(NamedLineCollectionBase);
public:
    NamedLineCollectionBase(const RenderGrid&, const String& name, GridPositionSide, bool nameIsAreaName);


    bool hasNamedLines() const;
    bool hasExplicitNamedLines() const;
    bool contains(unsigned line) const;
protected:

    void ensureInheritedNamedIndices();

    const Vector<unsigned>* m_namedLinesIndices { nullptr };
    const Vector<unsigned>* m_autoRepeatNamedLinesIndices { nullptr };
    const Vector<unsigned>* m_implicitNamedLinesIndices { nullptr };

    Vector<unsigned> m_inheritedNamedLinesIndices;

    unsigned m_insertionPoint { 0 };
    unsigned m_lastLine { 0 };
    unsigned m_autoRepeatTotalTracks { 0 };
    unsigned m_autoRepeatLines { 0 };
    unsigned m_autoRepeatTrackListLength { 0 };
    bool m_isSubgrid { false };
};

class NamedLineCollection : public NamedLineCollectionBase {
    WTF_MAKE_NONCOPYABLE(NamedLineCollection);
public:
    NamedLineCollection(const RenderGrid&, const String& name, GridPositionSide, bool nameIsAreaName = false);

    int firstPosition() const;

    unsigned lastLine() const;

private:
    int firstExplicitPosition() const;
};

// Class with all the code related to grid items positions resolution.
class GridPositionsResolver {
public:
    static GridPositionSide initialPositionSide(GridTrackSizingDirection);
    static GridPositionSide finalPositionSide(GridTrackSizingDirection);
    static unsigned spanSizeForAutoPlacedItem(const RenderBox&, GridTrackSizingDirection);
    static GridSpan resolveGridPositionsFromStyle(const RenderGrid& gridContainer, const RenderBox&, GridTrackSizingDirection);
    static unsigned explicitGridColumnCount(const RenderGrid&);
    static unsigned explicitGridRowCount(const RenderGrid&);
};

} // namespace WebCore
