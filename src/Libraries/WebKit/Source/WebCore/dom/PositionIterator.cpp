/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#include "PositionIterator.h"

#include "Editing.h"
#include "HTMLBodyElement.h"
#include "HTMLElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLNames.h"
#include "RenderBlockFlow.h"
#include "RenderBoxInlines.h"
#include "RenderFlexibleBox.h"
#include "RenderGrid.h"
#include "RenderText.h"

namespace WebCore {

using namespace HTMLNames;

PositionIterator::operator Position() const
{
    auto anchorNode = protectedNode();
    if (m_nodeAfterPositionInAnchor) {
        ASSERT(m_nodeAfterPositionInAnchor->parentNode() == anchorNode.get());
        // FIXME: This check is inadaquete because any ancestor could be ignored by editing
        if (positionBeforeOrAfterNodeIsCandidate(*anchorNode))
            return positionBeforeNode(anchorNode.get());
        return positionInParentBeforeNode(m_nodeAfterPositionInAnchor.get());
    }
    if (positionBeforeOrAfterNodeIsCandidate(*anchorNode))
        return atStartOfNode() ? positionBeforeNode(anchorNode.get()) : positionAfterNode(anchorNode.get());
    if (anchorNode->hasChildNodes())
        return lastPositionInOrAfterNode(anchorNode.get());
    return makeDeprecatedLegacyPosition(WTFMove(anchorNode), m_offsetInAnchor);
}

void PositionIterator::increment()
{
    if (!m_anchorNode)
        return;

    if (m_nodeAfterPositionInAnchor) {
        m_anchorNode = m_nodeAfterPositionInAnchor;
        m_nodeAfterPositionInAnchor = m_anchorNode->firstChild();
        m_offsetInAnchor = 0;
        return;
    }

    auto anchorNode = protectedNode();
    if (anchorNode->renderer() && !anchorNode->hasChildNodes() && m_offsetInAnchor < lastOffsetForEditing(*anchorNode))
        m_offsetInAnchor = Position::uncheckedNextOffset(anchorNode.get(), m_offsetInAnchor);
    else {
        m_nodeAfterPositionInAnchor = WTFMove(anchorNode);
        m_anchorNode = m_nodeAfterPositionInAnchor->parentNode();
        m_nodeAfterPositionInAnchor = m_nodeAfterPositionInAnchor->nextSibling();
        m_offsetInAnchor = 0;
    }
}

void PositionIterator::decrement()
{
    if (!m_anchorNode)
        return;

    if (m_nodeAfterPositionInAnchor) {
        m_anchorNode = m_nodeAfterPositionInAnchor->previousSibling();
        if (auto anchorNode = protectedNode()) {
            m_nodeAfterPositionInAnchor = nullptr;
            m_offsetInAnchor = anchorNode->hasChildNodes() ? 0 : lastOffsetForEditing(*anchorNode);
        } else {
            m_nodeAfterPositionInAnchor = m_nodeAfterPositionInAnchor->parentNode();
            m_anchorNode = m_nodeAfterPositionInAnchor->parentNode();
            m_offsetInAnchor = 0;
        }
        return;
    }
    
    if (m_anchorNode->hasChildNodes()) {
        m_anchorNode = m_anchorNode->lastChild();
        m_offsetInAnchor = m_anchorNode->hasChildNodes()? 0: lastOffsetForEditing(*m_anchorNode);
    } else {
        if (m_offsetInAnchor && m_anchorNode->renderer())
            m_offsetInAnchor = Position::uncheckedPreviousOffset(m_anchorNode.get(), m_offsetInAnchor);
        else {
            m_nodeAfterPositionInAnchor = m_anchorNode;
            m_anchorNode = m_anchorNode->parentNode();
        }
    }
}

bool PositionIterator::atStart() const
{
    if (!m_anchorNode)
        return true;
    if (m_anchorNode->parentNode())
        return false;
    return (!m_anchorNode->hasChildNodes() && !m_offsetInAnchor) || (m_nodeAfterPositionInAnchor && !m_nodeAfterPositionInAnchor->previousSibling());
}

bool PositionIterator::atEnd() const
{
    auto anchorNode = protectedNode();
    if (!anchorNode)
        return true;
    if (m_nodeAfterPositionInAnchor)
        return false;
    return !anchorNode->parentNode() && (anchorNode->hasChildNodes() || m_offsetInAnchor >= lastOffsetForEditing(*anchorNode));
}

bool PositionIterator::atStartOfNode() const
{
    if (!m_anchorNode)
        return true;
    if (!m_nodeAfterPositionInAnchor)
        return !m_anchorNode->hasChildNodes() && !m_offsetInAnchor;
    return !m_nodeAfterPositionInAnchor->previousSibling();
}

bool PositionIterator::atEndOfNode() const
{
    auto anchorNode = protectedNode();
    if (!anchorNode)
        return true;
    if (m_nodeAfterPositionInAnchor)
        return false;
    return anchorNode->hasChildNodes() || m_offsetInAnchor >= lastOffsetForEditing(*anchorNode);
}

// This function should be kept in sync with Position::isCandidate().
bool PositionIterator::isCandidate() const
{
    auto anchorNode = protectedNode();
    if (!anchorNode)
        return false;

    RenderObject* renderer = anchorNode->renderer();
    if (!renderer)
        return false;

    if (renderer->style().visibility() != Visibility::Visible)
        return false;

    if (renderer->isBR())
        return Position(*this).isCandidate();

    if (auto* renderText = dynamicDowncast<RenderText>(*renderer))
        return !Position::nodeIsUserSelectNone(anchorNode.get()) && renderText->containsCaretOffset(m_offsetInAnchor);

    if (positionBeforeOrAfterNodeIsCandidate(*anchorNode))
        return (atStartOfNode() || atEndOfNode()) && !Position::nodeIsUserSelectNone(anchorNode->parentNode());

    if (is<HTMLHtmlElement>(*anchorNode))
        return false;

    if (auto* block = dynamicDowncast<RenderBlock>(*renderer)) {
        if (is<RenderBlockFlow>(*block) || is<RenderGrid>(*block) || is<RenderFlexibleBox>(*block)) {
            if (block->logicalHeight() || is<HTMLBodyElement>(*anchorNode) || anchorNode->isRootEditableElement()) {
                if (!Position::hasRenderedNonAnonymousDescendantsWithHeight(*block))
                    return atStartOfNode() && !Position::nodeIsUserSelectNone(anchorNode.get());
                return anchorNode->hasEditableStyle() && !Position::nodeIsUserSelectNone(anchorNode.get()) && Position(*this).atEditingBoundary();
            }
            return false;
        }
    }

    return anchorNode->hasEditableStyle() && !Position::nodeIsUserSelectNone(anchorNode.get()) && Position(*this).atEditingBoundary();
}

} // namespace WebCore
