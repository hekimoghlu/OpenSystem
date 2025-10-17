/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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

#include "BoundaryPoint.h"
#include "CaretRectComputation.h"
#include "InlineIteratorBox.h"
#include "InlineIteratorLineBox.h"
#include "TextAffinity.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class LayoutUnit;
class Position;
class RenderObject;
class VisiblePosition;

class RenderedPosition {
public:
    RenderedPosition();
    explicit RenderedPosition(const VisiblePosition&);
    explicit RenderedPosition(const Position&, Affinity);
    bool isEquivalent(const RenderedPosition&) const;

    bool isNull() const { return !m_renderer; }
    InlineIterator::LineBoxIterator lineBox() const { return m_box ? m_box->lineBox() : InlineIterator::LineBoxIterator(); }
    InlineIterator::LeafBoxIterator box() const { return m_box; }
    unsigned offset() const { return m_offset; }

    unsigned char bidiLevelOnLeft() const;
    unsigned char bidiLevelOnRight() const;
    RenderedPosition leftBoundaryOfBidiRun(unsigned char bidiLevelOfRun) const;
    RenderedPosition rightBoundaryOfBidiRun(unsigned char bidiLevelOfRun) const;

    enum ShouldMatchBidiLevel { MatchBidiLevel, IgnoreBidiLevel };
    bool atLeftBoundaryOfBidiRun() const { return atLeftBoundaryOfBidiRun(IgnoreBidiLevel, 0); }
    bool atRightBoundaryOfBidiRun() const { return atRightBoundaryOfBidiRun(IgnoreBidiLevel, 0); }
    // The following two functions return true only if the current position is at the end of the bidi run
    // of the specified bidi embedding level.
    bool atLeftBoundaryOfBidiRun(unsigned char bidiLevelOfRun) const { return atLeftBoundaryOfBidiRun(MatchBidiLevel, bidiLevelOfRun); }
    bool atRightBoundaryOfBidiRun(unsigned char bidiLevelOfRun) const { return atRightBoundaryOfBidiRun(MatchBidiLevel, bidiLevelOfRun); }

    Position positionAtLeftBoundaryOfBiDiRun() const;
    Position positionAtRightBoundaryOfBiDiRun() const;

    bool atLeftmostOffsetInBox() const { return m_box && m_offset == m_box->leftmostCaretOffset(); }
    bool atRightmostOffsetInBox() const { return m_box && m_offset == m_box->rightmostCaretOffset(); }

    IntRect absoluteRect(CaretRectMode = CaretRectMode::Normal) const;

    std::optional<BoundaryPoint> boundaryPoint() const;

private:
    bool operator==(const RenderedPosition&) const { return false; }
    explicit RenderedPosition(const RenderObject*, InlineIterator::LeafBoxIterator, unsigned offset);

    InlineIterator::LeafBoxIterator previousLeafOnLine() const;
    InlineIterator::LeafBoxIterator nextLeafOnLine() const;
    bool atLeftBoundaryOfBidiRun(ShouldMatchBidiLevel, unsigned char bidiLevelOfRun) const;
    bool atRightBoundaryOfBidiRun(ShouldMatchBidiLevel, unsigned char bidiLevelOfRun) const;

    SingleThreadWeakPtr<const RenderObject> m_renderer;
    InlineIterator::LeafBoxIterator m_box;
    unsigned m_offset { 0 };

    mutable std::optional<InlineIterator::LeafBoxIterator> m_previousLeafOnLine;
    mutable std::optional<InlineIterator::LeafBoxIterator> m_nextLeafOnLine;
};

bool renderObjectContainsPosition(const RenderObject*, const Position&);

} // namespace WebCore
