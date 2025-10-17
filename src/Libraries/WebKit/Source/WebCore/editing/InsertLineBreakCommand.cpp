/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#include "InsertLineBreakCommand.h"

#include "Document.h"
#include "Editing.h"
#include "FrameSelection.h"
#include "HTMLBRElement.h"
#include "HTMLHRElement.h"
#include "HTMLNames.h"
#include "HTMLTableElement.h"
#include "LocalFrame.h"
#include "RenderElement.h"
#include "RenderStyleInlines.h"
#include "RenderText.h"
#include "Text.h"
#include "VisibleUnits.h"

namespace WebCore {

using namespace HTMLNames;

InsertLineBreakCommand::InsertLineBreakCommand(Ref<Document>&& document)
    : CompositeEditCommand(WTFMove(document))
{
}

bool InsertLineBreakCommand::preservesTypingStyle() const
{
    return true;
}

// Whether we should insert a break element or a '\n'.
bool InsertLineBreakCommand::shouldUseBreakElement(const Position& position)
{
    // An editing position like [input, 0] actually refers to the position before
    // the input element, and in that case we need to check the input element's
    // parent's renderer.
    auto node = position.parentAnchoredEquivalent().protectedDeprecatedNode();
    return node && node->renderer() && !node->renderer()->style().preserveNewline();
}

void InsertLineBreakCommand::doApply()
{
    deleteSelection();
    VisibleSelection selection = endingSelection();
    if (selection.isNoneOrOrphaned())
        return;
    
    VisiblePosition caret(selection.visibleStart());
    // FIXME: If the node is hidden, we should still be able to insert text. 
    // For now, we return to avoid a crash.  https://bugs.webkit.org/show_bug.cgi?id=40342
    if (caret.isNull())
        return;

    Position position(caret.deepEquivalent());

    position = positionAvoidingSpecialElementBoundary(position);
    position = positionOutsideTabSpan(position);

    if (!isEditablePosition(position))
        return;

    Ref document = protectedDocument();
    RefPtr<Node> nodeToInsert;
    if (shouldUseBreakElement(position))
        nodeToInsert = HTMLBRElement::create(document);
    else
        nodeToInsert = document->createTextNode("\n"_s);
    
    // FIXME: Need to merge text nodes when inserting just after or before text.
    document->updateLayoutIgnorePendingStylesheets();
    if (isEndOfParagraph(caret) && !lineBreakExistsAtVisiblePosition(caret)) {
        bool needExtraLineBreak = !is<HTMLHRElement>(*position.deprecatedNode()) && !is<HTMLTableElement>(*position.deprecatedNode());

        insertNodeAt(*nodeToInsert, position);
        
        if (needExtraLineBreak)
            insertNodeBefore(nodeToInsert->cloneNode(false), *nodeToInsert);
        
        VisiblePosition endingPosition(positionBeforeNode(nodeToInsert.get()));
        setEndingSelection(VisibleSelection(endingPosition, endingSelection().directionality()));
    } else if (position.deprecatedEditingOffset() <= caretMinOffset(*position.deprecatedNode())) {
        insertNodeAt(*nodeToInsert, position);
        
        // Insert an extra br or '\n' if the just inserted one collapsed.
        if (!isStartOfParagraph(positionBeforeNode(nodeToInsert.get())))
            insertNodeBefore(nodeToInsert->cloneNode(false), *nodeToInsert);
        
        setEndingSelection(VisibleSelection(positionInParentAfterNode(nodeToInsert.get()), Affinity::Downstream, endingSelection().directionality()));
    // If we're inserting after all of the rendered text in a text node, or into a non-text node,
    // a simple insertion is sufficient.
    } else if (position.deprecatedEditingOffset() >= caretMaxOffset(*position.deprecatedNode()) || !is<Text>(*position.deprecatedNode())) {
        insertNodeAt(*nodeToInsert, position);
        setEndingSelection(VisibleSelection(positionInParentAfterNode(nodeToInsert.get()), Affinity::Downstream, endingSelection().directionality()));
    } else if (RefPtr textNode = dynamicDowncast<Text>(*position.deprecatedNode())) {
        // Split a text node
        splitTextNode(*textNode, position.deprecatedEditingOffset());
        insertNodeBefore(*nodeToInsert, *textNode);
        Position endingPosition = firstPositionInNode(textNode.get());
        
        // Handle whitespace that occurs after the split
        document->updateLayoutIgnorePendingStylesheets();
        if (!endingPosition.isRenderedCharacter()) {
            Position positionBeforeTextNode(positionInParentBeforeNode(textNode.get()));
            // Clear out all whitespace and insert one non-breaking space
            deleteInsignificantTextDownstream(endingPosition);
            ASSERT(!textNode->renderer() || textNode->renderer()->style().collapseWhiteSpace());
            // Deleting insignificant whitespace will remove textNode if it contains nothing but insignificant whitespace.
            if (textNode->isConnected())
                insertTextIntoNode(*textNode, 0, nonBreakingSpaceString());
            else {
                auto nbspNode = document->createTextNode(String { nonBreakingSpaceString() });
                insertNodeAt(nbspNode.copyRef(), positionBeforeTextNode);
                endingPosition = firstPositionInNode(nbspNode.ptr());
            }
        }
        
        setEndingSelection(VisibleSelection(endingPosition, Affinity::Downstream, endingSelection().directionality()));
    }

    // Handle the case where there is a typing style.

    RefPtr<EditingStyle> typingStyle = document->selection().typingStyle();

    if (typingStyle && !typingStyle->isEmpty()) {
        // Apply the typing style to the inserted line break, so that if the selection
        // leaves and then comes back, new input will have the right style.
        // FIXME: We shouldn't always apply the typing style to the line break here,
        // see <rdar://problem/5794462>.
        applyStyle(typingStyle.get(), firstPositionInOrBeforeNode(nodeToInsert.get()), lastPositionInOrAfterNode(nodeToInsert.get()));
        // Even though this applyStyle operates on a Range, it still sets an endingSelection().
        // It tries to set a VisibleSelection around the content it operated on. So, that VisibleSelection
        // will either (a) select the line break we inserted, or it will (b) be a caret just 
        // before the line break (if the line break is at the end of a block it isn't selectable).
        // So, this next call sets the endingSelection() to a caret just after the line break 
        // that we inserted, or just before it if it's at the end of a block.
        setEndingSelection(endingSelection().visibleEnd());
    }

    rebalanceWhitespace();
}

}
