/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "RenderedPosition.h"

#include "CaretRectComputation.h"
#include "InlineRunAndOffset.h"
#include "VisiblePosition.h"

namespace WebCore {

static inline const RenderObject* rendererFromPosition(const Position& position)
{
    ASSERT(position.isNotNull());
    RefPtr<Node> rendererNode;
    switch (position.anchorType()) {
    case Position::PositionIsOffsetInAnchor:
        rendererNode = position.computeNodeAfterPosition();
        if (!rendererNode || !rendererNode->renderer())
            rendererNode = position.anchorNode()->lastChild();
        break;

    case Position::PositionIsBeforeAnchor:
    case Position::PositionIsAfterAnchor:
        break;

    case Position::PositionIsBeforeChildren:
        rendererNode = position.anchorNode()->firstChild();
        break;
    case Position::PositionIsAfterChildren:
        rendererNode = position.anchorNode()->lastChild();
        break;
    }
    if (!rendererNode || !rendererNode->renderer())
        rendererNode = position.anchorNode();
    return rendererNode->renderer();
}

RenderedPosition::RenderedPosition()
{
}

RenderedPosition::RenderedPosition(const RenderObject* renderer, InlineIterator::LeafBoxIterator box, unsigned offset)
    : m_renderer(renderer)
    , m_box(box)
    , m_offset(offset)
{
}

RenderedPosition::RenderedPosition(const VisiblePosition& position)
    : RenderedPosition(position.deepEquivalent(), position.affinity())
{
}

RenderedPosition::RenderedPosition(const Position& position, Affinity affinity)
{
    if (position.isNull())
        return;

    auto boxAndOffset = position.inlineBoxAndOffset(affinity);
    m_box = boxAndOffset.box;
    m_offset = boxAndOffset.offset;
    if (m_box)
        m_renderer = &m_box->renderer();
    else
        m_renderer = rendererFromPosition(position);
}

InlineIterator::LeafBoxIterator RenderedPosition::previousLeafOnLine() const
{
    if (!m_previousLeafOnLine)
        m_previousLeafOnLine = m_box->nextLineLeftwardOnLineIgnoringLineBreak();
    return *m_previousLeafOnLine;
}

InlineIterator::LeafBoxIterator RenderedPosition::nextLeafOnLine() const
{
    if (!m_nextLeafOnLine)
        m_nextLeafOnLine = m_box->nextLineRightwardOnLineIgnoringLineBreak();
    return *m_nextLeafOnLine;
}

bool RenderedPosition::isEquivalent(const RenderedPosition& other) const
{
    return (m_renderer == other.m_renderer && m_box == other.m_box && m_offset == other.m_offset)
        || (atLeftmostOffsetInBox() && other.atRightmostOffsetInBox() && previousLeafOnLine() == other.m_box)
        || (atRightmostOffsetInBox() && other.atLeftmostOffsetInBox() && nextLeafOnLine() == other.m_box);
}

unsigned char RenderedPosition::bidiLevelOnLeft() const
{
    auto box = atLeftmostOffsetInBox() ? previousLeafOnLine() : m_box;
    return box ? box->bidiLevel() : 0;
}

unsigned char RenderedPosition::bidiLevelOnRight() const
{
    auto box = atRightmostOffsetInBox() ? nextLeafOnLine() : m_box;
    return box ? box->bidiLevel() : 0;
}

RenderedPosition RenderedPosition::leftBoundaryOfBidiRun(unsigned char bidiLevelOfRun) const
{
    if (!m_box || bidiLevelOfRun > m_box->bidiLevel())
        return RenderedPosition();

    auto box = m_box;
    do {
        auto prev = box->nextLineLeftwardOnLineIgnoringLineBreak();
        if (!prev || prev->bidiLevel() < bidiLevelOfRun)
            return RenderedPosition(&box->renderer(), box, box->leftmostCaretOffset());
        box = prev;
    } while (box);

    ASSERT_NOT_REACHED();
    return RenderedPosition();
}

RenderedPosition RenderedPosition::rightBoundaryOfBidiRun(unsigned char bidiLevelOfRun) const
{
    if (!m_box || bidiLevelOfRun > m_box->bidiLevel())
        return RenderedPosition();

    auto box = m_box;
    do {
        auto next = box->nextLineRightwardOnLineIgnoringLineBreak();
        if (!next || next->bidiLevel() < bidiLevelOfRun)
            return RenderedPosition(&box->renderer(), box, box->rightmostCaretOffset());
        box = next;
    } while (box);

    ASSERT_NOT_REACHED();
    return RenderedPosition();
}

bool RenderedPosition::atLeftBoundaryOfBidiRun(ShouldMatchBidiLevel shouldMatchBidiLevel, unsigned char bidiLevelOfRun) const
{
    if (!m_box)
        return false;

    if (atLeftmostOffsetInBox()) {
        if (shouldMatchBidiLevel == IgnoreBidiLevel)
            return !previousLeafOnLine() || previousLeafOnLine()->bidiLevel() < m_box->bidiLevel();
        return m_box->bidiLevel() >= bidiLevelOfRun && (!previousLeafOnLine() || previousLeafOnLine()->bidiLevel() < bidiLevelOfRun);
    }

    if (atRightmostOffsetInBox()) {
        if (shouldMatchBidiLevel == IgnoreBidiLevel)
            return nextLeafOnLine() && m_box->bidiLevel() < nextLeafOnLine()->bidiLevel();
        return nextLeafOnLine() && m_box->bidiLevel() < bidiLevelOfRun && nextLeafOnLine()->bidiLevel() >= bidiLevelOfRun;
    }

    return false;
}

bool RenderedPosition::atRightBoundaryOfBidiRun(ShouldMatchBidiLevel shouldMatchBidiLevel, unsigned char bidiLevelOfRun) const
{
    if (!m_box)
        return false;

    if (atRightmostOffsetInBox()) {
        if (shouldMatchBidiLevel == IgnoreBidiLevel)
            return !nextLeafOnLine() || nextLeafOnLine()->bidiLevel() < m_box->bidiLevel();
        return m_box->bidiLevel() >= bidiLevelOfRun && (!nextLeafOnLine() || nextLeafOnLine()->bidiLevel() < bidiLevelOfRun);
    }

    if (atLeftmostOffsetInBox()) {
        if (shouldMatchBidiLevel == IgnoreBidiLevel)
            return previousLeafOnLine() && m_box->bidiLevel() < previousLeafOnLine()->bidiLevel();
        return previousLeafOnLine() && m_box->bidiLevel() < bidiLevelOfRun && previousLeafOnLine()->bidiLevel() >= bidiLevelOfRun;
    }

    return false;
}

Position RenderedPosition::positionAtLeftBoundaryOfBiDiRun() const
{
    ASSERT(atLeftBoundaryOfBidiRun());

    if (atLeftmostOffsetInBox())
        return makeDeprecatedLegacyPosition(m_renderer->protectedNode().get(), m_offset);

    return makeDeprecatedLegacyPosition(nextLeafOnLine()->renderer().protectedNode().get(), nextLeafOnLine()->leftmostCaretOffset());
}

Position RenderedPosition::positionAtRightBoundaryOfBiDiRun() const
{
    ASSERT(atRightBoundaryOfBidiRun());

    if (atRightmostOffsetInBox())
        return makeDeprecatedLegacyPosition(m_renderer->protectedNode().get(), m_offset);

    return makeDeprecatedLegacyPosition(previousLeafOnLine()->renderer().protectedNode().get(), previousLeafOnLine()->rightmostCaretOffset());
}

IntRect RenderedPosition::absoluteRect(CaretRectMode caretRectMode) const
{
    if (isNull())
        return IntRect();

    IntRect localRect = snappedIntRect(computeLocalCaretRect(*m_renderer, { m_box, m_offset }, caretRectMode));
    return localRect == IntRect() ? IntRect() : m_renderer->localToAbsoluteQuad(FloatRect(localRect)).enclosingBoundingBox();
}

std::optional<BoundaryPoint> RenderedPosition::boundaryPoint() const
{
    if (!m_box)
        return std::nullopt;

    RefPtr node = m_box->renderer().node();
    if (!node)
        return std::nullopt;

    return BoundaryPoint { *node, offset() };
}

bool renderObjectContainsPosition(const RenderObject* target, const Position& position)
{
    for (auto* renderer = rendererFromPosition(position); renderer && renderer->node(); renderer = renderer->parent()) {
        if (renderer == target)
            return true;
    }
    return false;
}

};
