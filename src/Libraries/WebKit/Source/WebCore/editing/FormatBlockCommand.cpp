/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#include "FormatBlockCommand.h"

#include "Document.h"
#include "Editing.h"
#include "Element.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "NodeName.h"
#include "VisibleUnits.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/RobinHoodHashSet.h>

namespace WebCore {

using namespace HTMLNames;

static RefPtr<Node> enclosingBlockToSplitTreeTo(Node* startNode);
static bool isElementForFormatBlock(const QualifiedName& tagName);

static inline bool isElementForFormatBlock(Node& node)
{
    auto* element = dynamicDowncast<Element>(node);
    return element && isElementForFormatBlock(element->tagQName());
}

FormatBlockCommand::FormatBlockCommand(Ref<Document>&& document, const QualifiedName& tagName)
    : ApplyBlockElementCommand(WTFMove(document), tagName)
    , m_didApply(false)
{
}

void FormatBlockCommand::formatSelection(const VisiblePosition& startOfSelection, const VisiblePosition& endOfSelection)
{
    if (!isElementForFormatBlock(tagName()))
        return;
    ApplyBlockElementCommand::formatSelection(startOfSelection, endOfSelection);
    m_didApply = true;
}

void FormatBlockCommand::formatRange(const Position& start, const Position& end, const Position& endOfSelection, RefPtr<Element>& blockNode)
{
    RefPtr nodeToSplitTo = enclosingBlockToSplitTreeTo(start.deprecatedNode());
    ASSERT(nodeToSplitTo);
    RefPtr<Node> outerBlock = (start.deprecatedNode() == nodeToSplitTo) ? start.deprecatedNode() : splitTreeToNode(*start.deprecatedNode(), *nodeToSplitTo);
    if (!outerBlock)
        return;

    RefPtr<Node> nodeAfterInsertionPosition = outerBlock;

    auto range = makeSimpleRange(start, endOfSelection);
    RefPtr refNode = enclosingBlockFlowElement(end);
    RefPtr root = editableRootForPosition(start);
    // Root is null for elements with contenteditable=false.
    if (!root || !refNode)
        return;
    if (isElementForFormatBlock(refNode->tagQName()) && start == startOfBlock(start)
        && (end == endOfBlock(end) || (range && isNodeVisiblyContainedWithin(*refNode, *range)))
        && refNode != root && !root->isDescendantOf(*refNode)) {
        // Already in a block element that only contains the current paragraph
        if (refNode->hasTagName(tagName()))
            return;
        nodeAfterInsertionPosition = WTFMove(refNode);
    }

    if (!blockNode) {
        // Create a new blockquote and insert it as a child of the root editable element. We accomplish
        // this by splitting all parents of the current paragraph up to that point.
        blockNode = createBlockElement();
        insertNodeBefore(*blockNode, *nodeAfterInsertionPosition);
    }

    Position lastParagraphInBlockNode = blockNode->lastChild() ? positionAfterNode(blockNode->lastChild()) : Position();
    bool wasEndOfParagraph = isEndOfParagraph(lastParagraphInBlockNode);

    moveParagraphWithClones(start, end, blockNode.get(), outerBlock.get());

    if (wasEndOfParagraph && lastParagraphInBlockNode.anchorNode()->isConnected()
        && !isEndOfParagraph(lastParagraphInBlockNode) && !isStartOfParagraph(lastParagraphInBlockNode))
        insertBlockPlaceholder(lastParagraphInBlockNode);
}

RefPtr<Element> FormatBlockCommand::elementForFormatBlockCommand(const std::optional<SimpleRange>& range)
{
    if (!range)
        return nullptr;

    RefPtr commonAncestor = commonInclusiveAncestor<ComposedTree>(*range);
    while (commonAncestor && !isElementForFormatBlock(*commonAncestor))
        commonAncestor = commonAncestor->parentNode();
    RefPtr commonAncestorElement = dynamicDowncast<Element>(commonAncestor);
    if (!commonAncestorElement)
        return nullptr;

    auto rootEditableElement = range->start.container->rootEditableElement();
    if (!rootEditableElement || commonAncestor->contains(rootEditableElement))
        return nullptr;

    return commonAncestorElement;
}

bool isElementForFormatBlock(const QualifiedName& tagName)
{
    using namespace ElementNames;

    switch (tagName.nodeName()) {
    case HTML::address:
    case HTML::article:
    case HTML::aside:
    case HTML::blockquote:
    case HTML::dd:
    case HTML::div:
    case HTML::dl:
    case HTML::dt:
    case HTML::footer:
    case HTML::h1:
    case HTML::h2:
    case HTML::h3:
    case HTML::h4:
    case HTML::h5:
    case HTML::h6:
    case HTML::header:
    case HTML::hgroup:
    case HTML::main:
    case HTML::nav:
    case HTML::p:
    case HTML::pre:
    case HTML::section:
        return true;
    default:
        break;
    }
    return false;
}

RefPtr<Node> enclosingBlockToSplitTreeTo(Node* startNode)
{
    RefPtr lastBlock = startNode;
    for (RefPtr n = startNode; n; n = n->parentNode()) {
        if (!n->hasEditableStyle())
            return lastBlock;
        if (isTableCell(*n) || n->hasTagName(bodyTag) || !n->parentNode() || !n->parentNode()->hasEditableStyle() || isElementForFormatBlock(*n))
            return n;
        if (isBlock(*n))
            lastBlock = n;
        if (isListHTMLElement(n.get())) {
            if (n->parentNode()->hasEditableStyle())
                return n->parentNode();
            return n;
        }
    }
    return lastBlock;
}

}
